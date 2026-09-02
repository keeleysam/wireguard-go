//go:build amd64 && gc && !purego

/* SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2017-2023 WireGuard LLC. All Rights Reserved.
 */

package device

import (
	"crypto/cipher"
	"os"

	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/sys/cpu"

	asmAEAD "github.com/tailscale/wireguard-go/tsasm/amd64/chacha20poly1305"
)

/*
shouldUseTSAsm reports whether the data-path AEAD should come from tsasm/amd64. Pure so a test can cover every combination; the CPU variables cannot be written.

Two disjoint populations reach tsasm, for opposite reasons.

A CPU with AVX512F, AVX512BW and BMI2 goes there because tsasm's fused kernel is faster than the AVX2 kernel it would otherwise get. Fusing Poly1305 into the cipher rounds is what buys it: on Intel the scalar MAC is 31 to 44 percent of the fused total, so hiding it under the rounds is most of the win. Measured per core against x/crypto's AVX2 at 1280-byte packets, Seal and Open: 1.21x and 1.37x on a Xeon D-2143IT, 1.20x and 1.43x on a Xeon 8375C, 1.23x and 1.50x on a Xeon 8488C, 1.11x and 1.43x on a Ryzen 9800X3D, 1.08x and 1.23x on a Xeon W-3245M. At 8920 bytes the range is 1.14x to 1.56x. Nothing measured slower than x/crypto on any of the five.

A CPU with SSSE3 and no AVX2 goes there because x/crypto has nothing for it at all: x/crypto gates its remaining amd64 kernel on HasSSSE3 && HasAVX2 && HasBMI2 and otherwise runs generic Go, which is 3.5x to 6.4x slower than assembly.

Everything in between, which is most amd64 hardware, keeps x/crypto's AVX2 kernel.

One thing those measurements do not cover, stated rather than left implicit. They time the AEAD on
an otherwise idle machine. Sustained 512-bit work puts Skylake-D and Cascade Lake-W into a lower
frequency licence, which on those parts can affect neighbouring cores, and tailscaled shares a
machine with whatever the operator is actually running. The judgement here is that the datapath win
is worth it: the kernel is 1.08x to 1.56x faster per core across five microarchitectures and never
slower, the licence effect is confined to the two oldest parts on that list, and newer AVX-512
hardware has progressively less of it. TS_WG_ASM=0 turns the whole thing off for anyone who
measures otherwise on their own workload.
*/
func shouldUseTSAsm(hasSSSE3, hasAVX2, hasBMI2, hasAVX512F, hasAVX512BW bool) bool {
	if hasAVX512F && hasAVX512BW && hasBMI2 {
		return true
	}
	return hasSSSE3 && !(hasAVX2 && hasBMI2)
}

var useTSAsm = shouldUseTSAsm(asmAEAD.Available(), cpu.X86.HasAVX2, cpu.X86.HasBMI2,
	cpu.X86.HasAVX512F, cpu.X86.HasAVX512BW)

/*
chacha20poly1305New returns a ChaCha20-Poly1305 AEAD. On amd64 CPUs with AVX-512, and on CPUs with SSSE3 but no AVX2, it uses the assembly kernels from tsasm/amd64/chacha20poly1305.

The cookie path (which uses the extended-nonce variant via chacha20poly1305.NewX) is left on the x/crypto path because it is not on the per-packet hot path.

As an escape hatch for hardware regressions or asm bugs, setting the environment variable TS_WG_ASM=0 forces the x/crypto implementation instead.
*/
func chacha20poly1305New(key []byte) (cipher.AEAD, error) {
	if !useTSAsm || os.Getenv("TS_WG_ASM") == "0" {
		return chacha20poly1305.New(key)
	}
	return asmAEAD.New(key)
}
