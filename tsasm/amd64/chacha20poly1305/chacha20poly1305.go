// SPDX-License-Identifier: BSD-3-Clause

//go:build amd64 && gc && !purego

/*
Package chacha20poly1305 provides a ChaCha20-Poly1305 AEAD backed by the SSE assembly kernel golang.org/x/crypto deleted in commit 7ee5970 ("chacha20poly1305: drop pre-AVX assembly impl", v0.52.0). It is used on amd64 CPUs with SSSE3 that cannot use x/crypto's AVX2 kernel.
*/
package chacha20poly1305

import (
	"crypto/cipher"
	"encoding/binary"
	"errors"
	"unsafe"

	"golang.org/x/sys/cpu"

	xchacha20poly1305 "golang.org/x/crypto/chacha20poly1305"
)

const (
	KeySize   = 32
	NonceSize = 12
	Overhead  = 16
)

// Available reports whether this CPU can run the SSSE3 kernel.
func Available() bool { return cpu.X86.HasSSSE3 }

/*
AvailableAVX512 reports whether this CPU can run the fused AVX-512 kernel.

The three features are read off the generated assembly rather than off the tier's name. AVX512BW
because the partial-group path stages through VMOVDQU8 and the mask moves are KMOVQ, and BMI2
because the Poly1305 chain multiplies with MULXQ. AVX512VL is not required, since the kernel issues
no 128-bit or 256-bit EVEX operation, and ADX is not required, since the carry chain uses ADCQ
rather than ADCXQ and ADOXQ. Gating on AVX512F alone would fault on a part with F but not BW.
*/
func AvailableAVX512() bool {
	return cpu.X86.HasAVX512F && cpu.X86.HasAVX512BW && cpu.X86.HasBMI2
}

//go:noescape
func chacha20Poly1305Open(dst []byte, key []uint32, src []byte, ad []byte) bool

//go:noescape
func chacha20Poly1305Seal(dst []byte, key []uint32, src []byte, ad []byte)

//go:noescape
func chacha20Poly1305OpenAVX512(dst []byte, key []uint32, src []byte, ad []byte) bool

//go:noescape
func chacha20Poly1305SealAVX512(dst []byte, key []uint32, src []byte, ad []byte)

/*
avx512MinSize is the AVX-512 kernels' only length requirement: at least one whole 256-byte block group, so the straight-line first group always has input. Anything past the last whole group goes through a staged partial-group path, so every length at or above this works and not only multiples of 256. That covers Tailscale's 1280, WireGuard's 1420 and 1440, and jumbo 8920.

Below it the SSSE3 kernel keeps its own small-buffer special cases, which are tuned for exactly that range and not worth duplicating.
*/
const avx512MinSize = 256

/*
avx512ShortSize is the length below which the AVX-512 tier hands the whole operation to x/crypto instead of running its own kernel.

The reason is that this tier is only ever dispatched to on a CPU that also has AVX2, so the thing it must beat at every length is x/crypto's AVX2 kernel, not our SSE tiers. Below one or two groups it does not: the fused pipeline's prologue and epilogue are a fixed cost that a short payload cannot amortise, and under avx512MinSize the fallback was our SSSE3 kernel, which on an AVX2 machine is far slower than the code the caller gave up. Measured on a Xeon D-2143IT, Seal against x/crypto's AVX2 ran 0.62x to 0.79x below 256 bytes and stayed under 1.00x to about 640.

768 is where all three AVX-512 parts measured cross over together, combined Seal and Open: 1.25x on Skylake-D, 1.16x on Zen 5 and 1.11x on Cascade Lake-W at 768, against 0.99x, 0.94x and 0.88x at 640. Delegating below it makes this tier no worse than the alternative at any length, which is a far easier property to argue than a win on average.
*/
const avx512ShortSize = 768

type aead struct {
	key    [KeySize]byte
	avx512 bool        // use the fused AVX-512 kernel where the payload is long enough
	short  cipher.AEAD // x/crypto, for payloads under avx512ShortSize; nil unless avx512
}

// New returns a ChaCha20-Poly1305 AEAD using the best assembly kernel this CPU can run: the fused
// AVX-512 one where it is available, otherwise the SSSE3 one.
func New(key []byte) (cipher.AEAD, error) {
	/*
	   Delegate to x/crypto rather than checking the key length here: its New also refuses
	   ChaCha20-Poly1305 under fips140=only, and this package is reached instead of x/crypto on the
	   CPUs it covers. The instance is kept rather than discarded, because the AVX-512 tier needs
	   one anyway for short payloads and New runs on every key rotation.
	*/
	short, err := xchacha20poly1305.New(key)
	if err != nil {
		return nil, err
	}
	/*
	   SSSE3 is checked before either tier is chosen, not just before the SSSE3 one. Every part with
	   AVX-512 has SSSE3, so this cannot fire today, but the AVX-512 tier still reaches
	   chacha20Poly1305Seal for payloads under avx512MinSize, and that kernel needs SSSE3. Checking
	   here means lowering avx512ShortSize, or dropping the short delegation, cannot turn into a
	   PSHUFB on a CPU whose SSSE3 support was never verified.
	*/
	if !Available() {
		return nil, errors.New("chacha20poly1305: CPU lacks SSSE3")
	}
	a := &aead{}
	copy(a.key[:], key)
	if AvailableAVX512() {
		a.avx512, a.short = true, short
	}
	return a, nil
}

/*
newAEADAVX512 pins the fused AVX-512 kernel regardless of what New would choose, for tests and
benchmarks that need to name a tier rather than take the best one. The feature requirements are on
AvailableAVX512 above.

Payloads below avx512ShortSize go to x/crypto, not to the SSSE3 kernel; see that constant.
*/
func newAEADAVX512(key []byte) (cipher.AEAD, error) {
	short, err := xchacha20poly1305.New(key)
	if err != nil {
		return nil, err
	}
	a := &aead{avx512: true, short: short}
	copy(a.key[:], key)
	return a, nil
}

func (a *aead) NonceSize() int { return NonceSize }
func (a *aead) Overhead() int  { return Overhead }

// setupState writes a ChaCha20 input matrix to state, per RFC 8439 §2.3.
// Copied from x/crypto's chacha20poly1305_amd64.go.
func setupState(state *[16]uint32, key *[KeySize]byte, nonce []byte) {
	state[0] = 0x61707865
	state[1] = 0x3320646e
	state[2] = 0x79622d32
	state[3] = 0x6b206574

	state[4] = binary.LittleEndian.Uint32(key[0:4])
	state[5] = binary.LittleEndian.Uint32(key[4:8])
	state[6] = binary.LittleEndian.Uint32(key[8:12])
	state[7] = binary.LittleEndian.Uint32(key[12:16])
	state[8] = binary.LittleEndian.Uint32(key[16:20])
	state[9] = binary.LittleEndian.Uint32(key[20:24])
	state[10] = binary.LittleEndian.Uint32(key[24:28])
	state[11] = binary.LittleEndian.Uint32(key[28:32])

	state[12] = 0
	state[13] = binary.LittleEndian.Uint32(nonce[0:4])
	state[14] = binary.LittleEndian.Uint32(nonce[4:8])
	state[15] = binary.LittleEndian.Uint32(nonce[8:12])
}

func (a *aead) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != NonceSize {
		panic("chacha20poly1305: bad nonce length passed to Seal")
	}
	if a.short != nil && len(plaintext) < avx512ShortSize {
		return a.short.Seal(dst, nonce, plaintext, additionalData)
	}
	if uint64(len(plaintext)) > (1<<38)-64 {
		panic("chacha20poly1305: plaintext too large")
	}

	var state [16]uint32
	setupState(&state, &a.key, nonce)

	ret, out := sliceForAppend(dst, len(plaintext)+Overhead)
	if inexactOverlap(out, plaintext) {
		panic("chacha20poly1305: invalid buffer overlap of output and input")
	}
	if anyOverlap(out, additionalData) {
		panic("chacha20poly1305: invalid buffer overlap of output and additional data")
	}
	if a.avx512 && len(plaintext) >= avx512MinSize {
		chacha20Poly1305SealAVX512(out[:], state[:], plaintext, additionalData)
	} else {
		chacha20Poly1305Seal(out[:], state[:], plaintext, additionalData)
	}
	return ret
}

var errOpen = errors.New("chacha20poly1305: message authentication failed")

func (a *aead) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != NonceSize {
		// Matches x/crypto: a wrong-size nonce is API misuse, not a decrypt failure.
		panic("chacha20poly1305: bad nonce length passed to Open")
	}
	if len(ciphertext) < Overhead {
		return nil, errOpen
	}
	if a.short != nil && len(ciphertext)-Overhead < avx512ShortSize {
		return a.short.Open(dst, nonce, ciphertext, additionalData)
	}
	if uint64(len(ciphertext)) > (1<<38)-48 {
		panic("chacha20poly1305: ciphertext too large")
	}

	var state [16]uint32
	setupState(&state, &a.key, nonce)

	ciphertext = ciphertext[:len(ciphertext)-Overhead]
	ret, out := sliceForAppend(dst, len(ciphertext))
	if inexactOverlap(out, ciphertext) {
		panic("chacha20poly1305: invalid buffer overlap of output and input")
	}
	if anyOverlap(out, additionalData) {
		panic("chacha20poly1305: invalid buffer overlap of output and additional data")
	}
	var ok bool
	if a.avx512 && len(ciphertext) >= avx512MinSize {
		ok = chacha20Poly1305OpenAVX512(out, state[:], ciphertext, additionalData)
	} else {
		ok = chacha20Poly1305Open(out, state[:], ciphertext, additionalData)
	}
	if !ok {
		for i := range out {
			out[i] = 0
		}
		return nil, errOpen
	}
	return ret, nil
}

func sliceForAppend(dst []byte, n int) (head, tail []byte) {
	if total := len(dst) + n; cap(dst) >= total {
		head = dst[:total]
	} else {
		head = make([]byte, total)
		copy(head, dst)
	}
	tail = head[len(dst):]
	return
}

// anyOverlap and inexactOverlap reproduce x/crypto/internal/alias, which cannot be
// imported from outside x/crypto: exact aliasing is fine, partial overlap is not.
func anyOverlap(x, y []byte) bool {
	return len(x) > 0 && len(y) > 0 &&
		uintptr(unsafe.Pointer(&x[0])) <= uintptr(unsafe.Pointer(&y[len(y)-1])) &&
		uintptr(unsafe.Pointer(&y[0])) <= uintptr(unsafe.Pointer(&x[len(x)-1]))
}

func inexactOverlap(x, y []byte) bool {
	if len(x) == 0 || len(y) == 0 || &x[0] == &y[0] {
		return false
	}
	return anyOverlap(x, y)
}
