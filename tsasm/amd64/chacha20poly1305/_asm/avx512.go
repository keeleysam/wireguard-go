// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A fused AVX-512 ChaCha20-Poly1305 for amd64, in a separate file so the SSE generator it
// sits beside is touched only by the tier branch added to main().
//
// Ported from the x/crypto fork this work was prototyped in; the vector half and the fusion
// schedule are unchanged, and only the Poly1305 multiply's name differs, because the suffix
// there tracked the AVX2 kernel that called it rather than the BMI2 it actually requires.
//
// The vector half reuses the layout proven by chacha20/_asm: one 128-bit lane per block, so
// a ZMM register holds one row of the ChaCha matrix for four blocks and four registers hold
// four complete block states. That is a group, 256 bytes.
//
// The scalar half reuses this package's existing Poly1305 emitters unchanged, so the
// carry-sensitive limb arithmetic is code that has already been fuzzed and audited; only the
// scheduling here is new.
//
// Poly1305 can only hash ciphertext that already exists, and within a multi-group iteration
// every round runs before every output, so a chunk's poly work has to hide under the next
// chunk's rounds. Single-group granularity is used because it leaves the least work
// unoverlapped at 1280 bytes, which is Tailscale's MTU and divides into 256 exactly. See
// chacha-poly1305-simd-implementation.md, phase A3b, for the measurements behind that.
package main

import (
	"flag"

	. "github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/ir"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

const (
	avx512BlocksPerGroup = 4
	avx512GroupBytes     = avx512BlocksPerGroup * 64 // 256
	avx512PolyBlocks     = avx512GroupBytes / 16     // 16 Poly1305 blocks per group
)

// Stack layout, BP-relative. r and s must sit at BP+0 and BP+16 because polyMulBMI2 and
// sealSSEFinalize already read them there.
var (
	avx512RSStore   = Mem{Base: BP}.Offset(0)   // r at +0, s at +16
	avx512KeyBlock  = Mem{Base: BP}.Offset(64)  // 64 bytes: block 0 of the counter-0 group
	avx512TailStore = Mem{Base: BP}.Offset(256) // 256 bytes: buffered partial tail
)

// avx512Group is one block-group's working state: four rows, four blocks per row.
type avx512Group struct {
	a, b, c, d VecVirtual
}

// chachaQR_AVX512 is ChaCha's quarter round. VPROLD rotates in one instruction and needs no
// scratch register, unlike chachaQR_AVX2 which spends two VPSHUFB plus four shifts and two
// xors on the same rotations and needs a temporary.
func chachaQR_AVX512(a, b, c, d VecVirtual) {
	VPADDD(b, a, a)
	VPXORD(a, d, d)
	VPROLD(U8(16), d, d)

	VPADDD(d, c, c)
	VPXORD(c, b, b)
	VPROLD(U8(12), b, b)

	VPADDD(b, a, a)
	VPXORD(a, d, d)
	VPROLD(U8(8), d, d)

	VPADDD(d, c, c)
	VPXORD(c, b, b)
	VPROLD(U8(7), b, b)
}

// diagonaliseAVX512 turns the column round into the diagonal round. VPSHUFD permutes dwords
// inside each 128-bit lane, and a lane is exactly one block's row.
func diagonaliseAVX512(b, c, d VecVirtual, forward bool) {
	if forward {
		VPSHUFD(U8(0x39), b, b)
		VPSHUFD(U8(0x4e), c, c)
		VPSHUFD(U8(0x93), d, d)
		return
	}
	VPSHUFD(U8(0x93), b, b)
	VPSHUFD(U8(0x4e), c, c)
	VPSHUFD(U8(0x39), d, d)
}

// avx512RoundClosures returns the group's twenty rounds as a list of emitters, so the caller
// can interleave another instruction stream between them.
func avx512RoundClosures(g avx512Group) []func() {
	var out []func()
	for r := 0; r < 10; r++ {
		out = append(out,
			func() { chachaQR_AVX512(g.a, g.b, g.c, g.d) },
			func() { diagonaliseAVX512(g.b, g.c, g.d, true) },
			func() { chachaQR_AVX512(g.a, g.b, g.c, g.d) },
			func() { diagonaliseAVX512(g.b, g.c, g.d, false) },
		)
	}
	return out
}

// avx512RoundsHashing emits one group's rounds with polyBlocks Poly1305 block absorptions
// spread evenly through them. hashAt gives the address of the i'th 16-byte block to absorb.
//
// Spreading is arithmetic over the two stream lengths rather than hand placement, which is
// the reason to want a generator here at all.
/*
avx512RoundClosuresBatch round-robins several groups' closure lists so consecutive emissions come from different groups. One group is a single dependency chain of vector work; n groups interleaved is n independent chains, which is what a wide out-of-order core needs to fill its vector units.

It is not free. Poly1305 can only hash ciphertext that already exists, and no output exists until a batch's rounds are all done, so widening the batch directly shrinks how much poly work can hide under rounds. At 1280 bytes, which is five groups, a four-group batch leaves four of the five groups' poly unoverlapped. Which way that trade lands is a question for measurement, which is why the batch size is a flag.
*/
func avx512RoundClosuresBatch(groups []avx512Group) []func() {
	per := make([][]func(), len(groups))
	for i, g := range groups {
		per[i] = avx512RoundClosures(g)
	}
	var out []func()
	for step := range per[0] {
		for i := range per {
			out = append(out, per[i][step])
		}
	}
	return out
}

// avx512NewBatch builds n groups on consecutive counters. Group 0 reuses dRow, so with n == 1
// this is exactly avx512NewGroup and the caller's counter advance is unchanged.
func avx512NewBatch(n int, initA, initB, initC, dRow VecVirtual, inc4 Mem) (work, init []avx512Group) {
	d := dRow
	for j := 0; j < n; j++ {
		if j > 0 {
			nd := ZMM()
			VPADDD(inc4, d, nd)
			d = nd
		}
		w, i := avx512NewGroup(initA, initB, initC, d)
		work = append(work, w)
		init = append(init, i)
	}
	return work, init
}

func avx512RoundsHashing(g avx512Group, polyBlocks int, hashAt func(i int) Mem) {
	avx512EmitHashing(avx512RoundClosures(g), polyBlocks, hashAt)
}

func avx512EmitHashing(rounds []func(), polyBlocks int, hashAt func(i int) Mem) {
	next := 0
	absorb := func(upto int) {
		for next < upto {
			polyAdd(hashAt(next))
			polyMulBMI2()
			next++
		}
	}
	for i, emit := range rounds {
		emit()
		absorb((i + 1) * polyBlocks / len(rounds))
	}
	absorb(polyBlocks)
}

/*
avx512MaskBytes builds one 64-byte lane mask per block for a run of count bytes, so a partial
group can be loaded, XORed and stored where it lies rather than staged through a scratch buffer
and copied a word at a time.

Block i covers bytes [64i, 64i+64), so its mask wants clamp(count-64i, 0, 64) low bits set.
BZHIQ zeroes every bit at and above the index it is handed, which is that mask exactly, and an
index of 64 or more leaves all sixty-four bits set, so a wholly-covered block needs no special
case. The clamp at the low end has to be explicit: a negative count read as an unsigned index
would set every bit rather than none.

Clobbers itr1, itr2 and t0, so this runs before any Poly1305 state is live. avo marks K0
restricted and never allocates it, which matters here, because as a predicate operand K0 encodes
"no mask" whatever its contents, so a store predicated on it would write the whole register.
*/
func avx512MaskBytes(count Register) [avx512BlocksPerGroup]OpmaskVirtual {
	var out [avx512BlocksPerGroup]OpmaskVirtual
	XORQ(itr2, itr2)
	for i := 0; i < avx512BlocksPerGroup; i++ {
		MOVQ(count, itr1)
		if i > 0 {
			SUBQ(U32(uint64(i)*64), itr1)
			CMOVQLT(itr2, itr1)
		}
		MOVQ(I32(-1), t0)
		BZHIQ(itr1, t0, t0)
		k := K()
		KMOVQ(t0, k)
		out[i] = k
	}
	return out
}

/*
sealAVX512MaskedPair emits the final two groups together: one whole group, then a partial group
of inl bytes where 0 < inl < 256.

Pairing them is the whole point. A partial group emitted on its own runs a full group's rounds
with nothing to interleave against, which measured as costing more than an entire extra whole
group on every machine tested, from 1.33x on Sapphire Rapids to 1.78x on Cascade Lake. Riding
alongside a whole group gives those rounds the same company every other group gets. Masking is
what makes the pairing expressible: the partial group is loaded, XORed and stored in registers
under a k-mask instead of being copied through a staging buffer a word at a time.

Poly1305 still reads the partial group from the zeroed staging buffer, because it hashes
ceil(inl/16)*16 bytes and everything past inl has to be zero for the final block's padding. A
masked store into a pre-zeroed buffer produces exactly that padding at no cost.

The backlog of whole groups written earlier is hashed under these rounds. This batch's own two
groups then have no rounds left to hide under, which is inherent to encrypt-then-MAC.

Pointers are deliberately not advanced: this is the end of the input.
*/
func sealAVX512MaskedPair(initA, initB, initC, dRow VecVirtual, inc4 Mem, backlogGroups int) {
	Comment("One lane mask per block, from the bytes that remain")
	masks := avx512MaskBytes(inl)

	Comment("Zero the staging buffer: the final partial block's padding comes from it")
	zero := ZMM()
	VPXORD(zero, zero, zero)
	for i := 0; i < avx512BlocksPerGroup; i++ {
		VMOVDQU32(zero, avx512TailStore.Offset(i*64))
	}

	work, init := avx512NewBatch(2, initA, initB, initC, dRow, inc4)
	Comment("Both groups' rounds interleaved, hashing the backlog behind them")
	avx512EmitHashing(avx512RoundClosuresBatch(work), backlogGroups*avx512PolyBlocks, func(i int) Mem {
		return Mem{Base: oup}.Offset(-backlogGroups*avx512GroupBytes + i*16)
	})

	for j := 0; j < 2; j++ {
		avx512AddInit(work[j], init[j])
		blocks := avx512Transpose(work[j])
		for i := 0; i < avx512BlocksPerGroup; i++ {
			at := j*avx512GroupBytes + i*64
			if j == 0 {
				VPXORD(Mem{Base: inp}.Offset(at), blocks[i], blocks[i])
				VMOVDQU32(blocks[i], Mem{Base: oup}.Offset(at))
				continue
			}
			src := ZMM()
			VMOVDQU8_Z(Mem{Base: inp}.Offset(at), masks[i], src)
			VPXORD(src, blocks[i], blocks[i])
			VMOVDQU8(blocks[i], masks[i], Mem{Base: oup}.Offset(at))
			VMOVDQU8(blocks[i], masks[i], avx512TailStore.Offset(i*64))
		}
	}

	Comment("Hash this batch: the whole group in place, then the partial one from the buffer")
	for i := 0; i < avx512PolyBlocks; i++ {
		polyAdd(Mem{Base: oup}.Offset(i * 16))
		polyMulBMI2()
	}
	avx512HashBuffer("sealMaskedPair", inl)

	// The caller writes the tag at oup, so oup has to end up just past the ciphertext even
	// though nothing reads it after this. Both hashing steps above address off the
	// un-advanced pointer, so this has to come last.
	Comment("Advance past the whole group and the partial one, so the tag lands after them")
	ADDQ(U32(avx512GroupBytes), oup)
	ADDQ(inl, oup)
	ADDQ(U32(avx512GroupBytes), inp)
	ADDQ(inl, inp)
}

/*
avx512Groups is how many independent groups the steady loop keeps in flight. One is the original schedule, which maximises Poly1305 overlap and gives the vector units a single chain. Higher values trade overlap for chains.

The default is the value the checked-in kernel is generated with, deliberately, so that regenerating without arguments reproduces what ships. It used to default to 1 while the checked-in file was built with 2, which made the committed assembly unreproducible from its own //go:generate line and was caught only by running the gated regeneration test.
*/
var avx512Groups = 2

var avx512GroupsFlag = flag.Int("groups", 2, "AVX-512 groups in flight per batch: 1, 2 or 4")

// avx512Transpose exchanges 128-bit lanes so each block's sixteen words land contiguously in
// one register, and returns the four per-block registers in block order.
//
// Going in, register a holds row 0 for blocks 0 to 3, so block j's keystream is lane j of a,
// b, c and d. VSHUFI64X2 takes its low two output lanes from src1 and its high two from
// src2, which is where the 0x44/0xEE then 0x88/0xDD immediates come from.
func avx512Transpose(g avx512Group) [4]VecVirtual {
	ab0, ab1, cd0, cd1 := ZMM(), ZMM(), ZMM(), ZMM()
	VSHUFI64X2(U8(0x44), g.b, g.a, ab0) // a0 a1 b0 b1
	VSHUFI64X2(U8(0xee), g.b, g.a, ab1) // a2 a3 b2 b3
	VSHUFI64X2(U8(0x44), g.d, g.c, cd0) // c0 c1 d0 d1
	VSHUFI64X2(U8(0xee), g.d, g.c, cd1) // c2 c3 d2 d3

	var blocks [4]VecVirtual
	for i, sel := range []struct {
		imm    uint8
		lo, hi VecVirtual
	}{
		{0x88, ab0, cd0},
		{0xdd, ab0, cd0},
		{0x88, ab1, cd1},
		{0xdd, ab1, cd1},
	} {
		blocks[i] = ZMM()
		VSHUFI64X2(U8(sel.imm), sel.hi, sel.lo, blocks[i])
	}
	return blocks
}

// avx512AddInit adds the original state back, which ChaCha requires before the keystream is
// usable.
func avx512AddInit(work, init avx512Group) {
	VPADDD(init.a, work.a, work.a)
	VPADDD(init.b, work.b, work.b)
	VPADDD(init.c, work.c, work.c)
	VPADDD(init.d, work.d, work.d)
}

// avx512NewGroup builds a group's initial state from the sixteen-word ChaCha state at keyp,
// with dRow supplying row 3 so the caller controls the block counters.
func avx512NewGroup(initA, initB, initC, dRow VecVirtual) (work, init avx512Group) {
	init = avx512Group{a: initA, b: initB, c: initC, d: dRow}
	work = avx512Group{a: ZMM(), b: ZMM(), c: ZMM(), d: ZMM()}
	VMOVDQA32(initA, work.a)
	VMOVDQA32(initB, work.b)
	VMOVDQA32(initC, work.c)
	VMOVDQA32(dRow, work.d)
	return work, init
}

/*
Both entry points use Implement rather than TEXT. Implement takes the signature from the Go
declaration in chacha20poly1305.go, which Package() resolved, so the argument offsets and frame
size are derived from the one authoritative declaration. Passing a signature string to TEXT instead
would put the same prototype in two files with nothing checking they agree, and a later edit to
either one would silently move the arguments. It is also what the SSSE3 generator beside this does.
*/
func chacha20Poly1305SealAVX512() {
	Implement("chacha20Poly1305SealAVX512")
	Attributes(0)
	Doc("chacha20Poly1305SealAVX512 seals with a fused AVX-512 ChaCha20 and scalar Poly1305.",
		"len(src) must be at least one whole 256-byte group; the caller sends anything shorter to the SSE kernels.")
	initA, initB, initC, dRow := avx512SetupState()
	inc4 := avx512Inc4_DATA()

	// One group of ciphertext is always written before any of it can be hashed, so the first
	// group carries no Poly1305 work and the steady loop then hashes the group behind it.
	// The caller guarantees at least one whole group, so the first group always runs; the
	// steady loop runs once per group after that and may run zero times. The epilogue hashes
	// the last full group, which has no rounds left to hide under, and the tail below handles
	// any bytes past it, so no alignment beyond the one-group minimum is required.
	// encryptBatch emits n groups with their rounds interleaved, hashing polyBlocks of
	// already-written output starting hashBack bytes behind the output pointer. With n == 1
	// and hashBack one group it is the original single-group schedule exactly.
	encryptBatch := func(n, polyBlocks, hashBack int) {
		work, init := avx512NewBatch(n, initA, initB, initC, dRow, inc4)
		hashAt := func(i int) Mem {
			return Mem{Base: oup}.Offset(-hashBack + i*16)
		}
		rounds := avx512RoundClosuresBatch(work)
		if polyBlocks > 0 {
			avx512EmitHashing(rounds, polyBlocks, hashAt)
		} else {
			for _, emit := range rounds {
				emit()
			}
		}
		for j := 0; j < n; j++ {
			avx512AddInit(work[j], init[j])
			blocks := avx512Transpose(work[j])
			for i := 0; i < avx512BlocksPerGroup; i++ {
				at := j*avx512GroupBytes + i*64
				VPXORD(Mem{Base: inp}.Offset(at), blocks[i], blocks[i])
				VMOVDQU32(blocks[i], Mem{Base: oup}.Offset(at))
			}
		}
		VPADDD(inc4, init[n-1].d, dRow)
		ADDQ(U32(n*avx512GroupBytes), inp)
		ADDQ(U32(n*avx512GroupBytes), oup)
		SUBQ(U32(n*avx512GroupBytes), inl)
	}

	hashBacklog := func(groups int) {
		for i := 0; i < groups*avx512PolyBlocks; i++ {
			polyAdd(Mem{Base: oup}.Offset(-groups*avx512GroupBytes + i*16))
			polyMulBMI2()
		}
	}

	if n := avx512Groups; n > 1 {
		batchBytes := n * avx512GroupBytes

		Comment("Too short for a whole batch: take the single-group schedule instead")
		CMPQ(inl, U32(batchBytes))
		JB(LabelRef("sealAVX512Single"))

		Comment("First batch: nothing has been written yet, so there is nothing to hash")
		encryptBatch(n, 0, 0)

		Comment("Steady state: hash the batch behind this one under this batch's rounds")
		Label("sealAVX512BatchLoop")
		CMPQ(inl, U32(batchBytes))
		JB(LabelRef("sealAVX512DrainEntry"))
		encryptBatch(n, n*avx512PolyBlocks, batchBytes)
		JMP(LabelRef("sealAVX512BatchLoop"))

		// The backlog is exactly n groups here. Hashing one group per single-group iteration
		// keeps it at n, so the overlap continues for whatever whole groups are left.
		//
		// One whole group is held back rather than consumed greedily, so that a partial group
		// has a full one to ride along with. That is what sealAVX512MaskedPair needs, and a
		// partial group emitted alone costs more than an entire extra whole group.
		Comment("Whole groups left over, still hashing a batch behind, keeping one in reserve")
		//
		// The remainder is tested once, here, rather than after the loop, and the two cases get
		// separate contiguous paths. Sharing one path and branching at the end measured as a
		// flat 215ns penalty on every exact-multiple length on Zen 5, on a hot loop that was
		// byte-for-byte identical: the cost was instruction fetch, from the common path having
		// to jump forward over the whole masked-pair block. Duplicating the drain loop costs
		// code size and buys that back.
		Comment("Whole groups left over. An exact multiple keeps the old path untouched.")
		Label("sealAVX512DrainEntry")
		MOVQ(inl, itr1)
		ANDQ(U32(avx512GroupBytes-1), itr1)
		JNZ(LabelRef("sealAVX512DrainReserve"))

		/*
		   Leave the last whole group for the catching-up block below rather than draining
		   everything here. That block hashes the entire backlog under one group's rounds instead of
		   one group of it, which halves what the epilogue then has to hash with nothing to hide it
		   under.

		   The loop only exists when a batch is wider than two groups. Control reaches here from the
		   steady loop's exit, so inl is already below batchBytes, and the loop requires inl to be
		   at least two groups: at n == 2 those are the same 512 bytes and the body is unreachable.
		   Emitting it anyway cost about 870 lines of assembly that could never execute.
		*/
		Label("sealAVX512Drain")
		if n > 2 {
			CMPQ(inl, U32(2*avx512GroupBytes))
			JB(LabelRef("sealAVX512DrainHash"))
			encryptBatch(1, avx512PolyBlocks, batchBytes)
			JMP(LabelRef("sealAVX512Drain"))
		}

		// One whole group may remain, because the loop above stops one short. Ciphering it
		// here with the whole backlog spread under its rounds leaves only this group's own
		// blocks for the epilogue, halving the exposed hash. Lengths that are an exact
		// multiple of the group size reach the epilogue with nothing else to hide under, and
		// 1280, Tailscale's MTU, is one of them.
		Label("sealAVX512DrainHash")
		CMPQ(inl, U32(avx512GroupBytes))
		JB(LabelRef("sealAVX512DrainHashAll"))
		Comment("One whole group left: cipher it while hashing the whole backlog")
		encryptBatch(1, n*avx512PolyBlocks, batchBytes)
		hashBacklog(1)
		JMP(LabelRef("sealAVX512Reduce"))

		Comment("Nothing left to cipher: hash the backlog with no rounds to hide it under")
		Label("sealAVX512DrainHashAll")
		hashBacklog(n)
		JMP(LabelRef("sealAVX512Reduce"))

		// Same reasoning as sealAVX512Drain above: unreachable at n == 2, so not emitted there.
		Comment("A partial group follows, so keep one whole group back to pair it with")
		Label("sealAVX512DrainReserve")
		if n > 2 {
			CMPQ(inl, U32(2*avx512GroupBytes))
			JB(LabelRef("sealAVX512PairOrTail"))
			encryptBatch(1, avx512PolyBlocks, batchBytes)
			JMP(LabelRef("sealAVX512DrainReserve"))
		}

		Label("sealAVX512PairOrTail")
		CMPQ(inl, U32(avx512GroupBytes))
		JB(LabelRef("sealAVX512SmallTail"))
		SUBQ(U32(avx512GroupBytes), inl)
		sealAVX512MaskedPair(initA, initB, initC, dRow, inc4, n)
		JMP(LabelRef("sealAVX512Reduce"))

		Comment("Not even one whole group left to pair with: hash the backlog and stage the tail")
		Label("sealAVX512SmallTail")
		hashBacklog(n)
		JMP(LabelRef("sealAVX512Tail"))

		Comment("Single-group schedule, for inputs shorter than one batch")
		Label("sealAVX512Single")
		encryptBatch(1, 0, 0)
		Label("sealAVX512SingleLoop")
		CMPQ(inl, U32(avx512GroupBytes))
		JB(LabelRef("sealAVX512SingleHash"))
		encryptBatch(1, avx512PolyBlocks, avx512GroupBytes)
		JMP(LabelRef("sealAVX512SingleLoop"))
		Label("sealAVX512SingleHash")
		hashBacklog(1)

		Label("sealAVX512Tail")
	} else {
		Comment("First group: nothing has been written yet, so there is nothing to hash")
		encryptBatch(1, 0, 0)

		Comment("Steady state: hash the previous group under this group's rounds")
		Label("sealAVX512Loop")
		CMPQ(inl, U32(avx512GroupBytes))
		JB(LabelRef("sealAVX512Epilogue"))
		encryptBatch(1, avx512PolyBlocks, avx512GroupBytes)
		JMP(LabelRef("sealAVX512Loop"))

		Comment("Hash the final full group, with no rounds left to hide it under")
		Label("sealAVX512Epilogue")
		hashBacklog(1)
	}

	Comment("Any bytes past the last whole group")
	TESTQ(inl, inl)
	JZ(LabelRef("sealAVX512Reduce"))
	avx512Tail("seal", initA, initB, initC, dRow, true)

	Label("sealAVX512Reduce")
	avx512Reduce()

	Comment("Store the tag at the end of the message")
	MOVQ(acc0, Mem{Base: oup}.Offset(0))
	MOVQ(acc1, Mem{Base: oup}.Offset(8))
	VZEROUPPER()
	RET()
}

// The three counter constants are memoised the same way the existing generator memoises its
// own DATA symbols: both directions ask for them, and emitting a GLOBL twice produces
// "overlapping DATA entry" at assembly time.
var (
	avx512IncMask_DATA_ptr *Mem
	avx512Inc4_DATA_ptr    *Mem
	avx512Inc1_DATA_ptr    *Mem
)

func avx512IncMask_DATA() Mem {
	if avx512IncMask_DATA_ptr != nil {
		return *avx512IncMask_DATA_ptr
	}
	g := GLOBL(ThatPeskyUnicodeDot+"avx512IncMask", NOPTR|RODATA)
	avx512IncMask_DATA_ptr = &g
	for lane := 0; lane < 4; lane++ {
		DATA(16*lane, U32(uint32(lane)))
		DATA(16*lane+4, U32(0))
		DATA(16*lane+8, U32(0))
		DATA(16*lane+12, U32(0))
	}
	return g
}

func avx512Inc4_DATA() Mem {
	if avx512Inc4_DATA_ptr != nil {
		return *avx512Inc4_DATA_ptr
	}
	g := GLOBL(ThatPeskyUnicodeDot+"avx512Inc4", NOPTR|RODATA)
	avx512Inc4_DATA_ptr = &g
	for lane := 0; lane < 4; lane++ {
		DATA(16*lane, U32(avx512BlocksPerGroup))
		DATA(16*lane+4, U32(0))
		DATA(16*lane+8, U32(0))
		DATA(16*lane+12, U32(0))
	}
	return g
}

func avx512Inc1_DATA() Mem {
	if avx512Inc1_DATA_ptr != nil {
		return *avx512Inc1_DATA_ptr
	}
	g := GLOBL(ThatPeskyUnicodeDot+"avx512Inc1", NOPTR|RODATA)
	avx512Inc1_DATA_ptr = &g
	for lane := 0; lane < 4; lane++ {
		DATA(16*lane, U32(1))
		DATA(16*lane+4, U32(0))
		DATA(16*lane+8, U32(0))
		DATA(16*lane+12, U32(0))
	}
	return g
}

// avx512SetupState emits the shared prologue for both directions: align BP, broadcast the
// three constant rows, build row 3 with the per-lane counter offsets, derive the Poly1305 key
// from block 0 of the counter-0 group, and advance row 3 so the payload starts at counter 1.
//
// Returns the three shared init rows and the row-3 register the caller advances per group.
func avx512SetupState() (initA, initB, initC, dRow VecVirtual) {
	/*
	   The frame is declared here rather than at each entry point because both directions need
	   exactly this layout and neither should be able to drift from it. Its size is fixed by the
	   BP-relative map above: 256 bytes of buffered partial tail ending at 512, rounded up.
	*/
	AllocLocal(768)
	MOVQ(RSP, RBP)
	ADDQ(Imm(32), RBP)
	ANDQ(I32(-32), RBP)

	Load(Param("dst").Base(), oup)
	Load(Param("key").Base(), keyp)
	Load(Param("src").Base(), inp)
	Load(Param("src").Len(), inl)
	Load(Param("ad").Base(), adp)

	initA, initB, initC, dRow = ZMM(), ZMM(), ZMM(), ZMM()
	VBROADCASTI32X4(chacha20Constants_DATA(), initA)
	VBROADCASTI32X4(Mem{Base: keyp}.Offset(16), initB)
	VBROADCASTI32X4(Mem{Base: keyp}.Offset(32), initC)
	VBROADCASTI32X4(Mem{Base: keyp}.Offset(48), dRow)
	VPADDD(avx512IncMask_DATA(), dRow, dRow)

	Comment("Poly1305 key from block 0 of the counter-0 group")
	work, init := avx512NewGroup(initA, initB, initC, dRow)
	for _, emit := range avx512RoundClosures(work) {
		emit()
	}
	avx512AddInit(work, init)
	blocks := avx512Transpose(work)
	VMOVDQU32(blocks[0], avx512KeyBlock)
	rKey, sKey := XMM(), XMM()
	VMOVDQU(avx512KeyBlock, rKey)
	VPAND(polyClampMask_DATA(), rKey, rKey)
	VMOVDQU(rKey, avx512RSStore)
	VMOVDQU(avx512KeyBlock.Offset(16), sKey)
	VMOVDQU(sKey, avx512RSStore.Offset(16))

	// The payload starts at counter 1, so move row 3 from counters 0-3 to counters 1-4.
	VPADDD(avx512Inc1_DATA(), dRow, dRow)

	Comment("Zero the Poly1305 accumulator, then hash the additional data")
	XORQ(acc0, acc0)
	XORQ(acc1, acc1)
	XORQ(acc2, acc2)
	MOVQ(NewParamAddr("ad_len", 80), itr2)
	CALL(LabelRef("polyHashADInternal<>(SB)"))

	return initA, initB, initC, dRow
}

// avx512Reduce emits the shared tail: fold in the two lengths, reduce mod 2^130-5 and add s.
func avx512Reduce() {
	Comment("Hash in the buffer lengths")
	ADDQ(NewParamAddr("ad_len", 80), acc0)
	ADCQ(NewParamAddr("src_len", 56), acc1)
	ADCQ(Imm(1), acc2)
	polyMul()

	Comment("Final reduce")
	MOVQ(acc0, t0)
	MOVQ(acc1, t1)
	MOVQ(acc2, t2)
	SUBQ(I8(-5), acc0)
	SBBQ(I8(-1), acc1)
	SBBQ(Imm(3), acc2)
	CMOVQCS(t0, acc0)
	CMOVQCS(t1, acc1)
	CMOVQCS(t2, acc2)

	Comment("Add in the \"s\" part of the key")
	ADDQ(avx512RSStore.Offset(16), acc0)
	ADCQ(avx512RSStore.Offset(24), acc1)
}

// chacha20Poly1305OpenAVX512 is the decrypting direction, and it is structurally simpler than
// the sealing one.
//
// Seal must hash ciphertext it has not produced yet, which forces a software pipeline: a
// group's Poly1305 work can only hide under the *next* group's rounds, so the first group
// carries no hashing and the last group's hashing has nothing to hide under. Open's MAC reads
// the input, which is already in memory and independent of the keystream, so every group
// hashes its own sixteen blocks under its own rounds. There is no prologue, no epilogue and
// no lag, and every group gets full overlap.
//
// The kernel decrypts before the tag is known to be correct, which is the posture the AVX2
// kernel beside it already ships; the Go caller zeroes the output buffer when this returns
// false.
func chacha20Poly1305OpenAVX512() {
	Implement("chacha20Poly1305OpenAVX512")
	Attributes(0)
	Doc("chacha20Poly1305OpenAVX512 opens with a fused AVX-512 ChaCha20 and scalar Poly1305.",
		"len(src) must be at least one whole 256-byte group; the caller sends anything shorter to the SSE kernels.",
		"Returns whether the tag authenticated. The caller must zero dst when it did not.")

	initA, initB, initC, dRow := avx512SetupState()
	inc4 := avx512Inc4_DATA()

	/*
		openBatch emits n groups with their rounds interleaved, each hashing its own sixteen
		blocks of input. Unlike Seal there is no lag to manage: the MAC reads the input, which
		already exists, so a batch needs no prologue hashing nothing and no epilogue with nothing
		to hide under. With n == 1 this is the original single-group schedule exactly.

		All n groups hash before any of them XORs, so decrypting in place stays safe.
	*/
	openBatch := func(n int) {
		work, init := avx512NewBatch(n, initA, initB, initC, dRow, inc4)
		rounds := avx512RoundClosuresBatch(work)
		avx512EmitHashing(rounds, n*avx512PolyBlocks, func(i int) Mem {
			return Mem{Base: inp}.Offset(i * 16)
		})
		for j := 0; j < n; j++ {
			avx512AddInit(work[j], init[j])
			blocks := avx512Transpose(work[j])
			for i := 0; i < avx512BlocksPerGroup; i++ {
				at := j*avx512GroupBytes + i*64
				VPXORD(Mem{Base: inp}.Offset(at), blocks[i], blocks[i])
				VMOVDQU32(blocks[i], Mem{Base: oup}.Offset(at))
			}
		}
		VPADDD(inc4, init[n-1].d, dRow)
		ADDQ(U32(n*avx512GroupBytes), inp)
		ADDQ(U32(n*avx512GroupBytes), oup)
		SUBQ(U32(n*avx512GroupBytes), inl)
	}

	if n := avx512Groups; n > 1 {
		Comment("Batches of groups, each hashing its own ciphertext under its own rounds")
		Label("openAVX512BatchLoop")
		CMPQ(inl, U32(n*avx512GroupBytes))
		JB(LabelRef("openAVX512Single"))
		openBatch(n)
		JMP(LabelRef("openAVX512BatchLoop"))

		Comment("Whole groups left over, fewer than one batch")
		Label("openAVX512Single")
	} else {
		Comment("Every group hashes its own ciphertext under its own rounds")
	}

	Label("openAVX512Loop")
	CMPQ(inl, U32(avx512GroupBytes))
	JB(LabelRef("openAVX512Tail"))
	openBatch(1)
	JMP(LabelRef("openAVX512Loop"))

	// The loop exits here rather than straight to the finalize label. Jumping past this
	// point would make the tail unreachable, which is exactly the bug this replaced: every
	// length that was not a multiple of 256 skipped its final partial group entirely, so
	// those bytes were neither decrypted nor hashed.
	Label("openAVX512Tail")
	Comment("Any bytes past the last whole group")
	TESTQ(inl, inl)
	JZ(LabelRef("openAVX512Finalize"))
	avx512Tail("open", initA, initB, initC, dRow, false)

	Label("openAVX512Finalize")
	avx512Reduce()

	Comment("Constant time compare against the tag, which sits just past the ciphertext")
	XORQ(RAX, RAX)
	MOVQ(U32(1), RDX)
	XORQ(Mem{Base: inp}.Offset(0*8), acc0)
	XORQ(Mem{Base: inp}.Offset(1*8), acc1)
	ORQ(acc1, acc0)
	CMOVQEQ(RDX, RAX)

	Comment("Return true iff the tags are equal")
	VZEROUPPER()
	// avo has no direct form for a bool return, so emit the store by hand, as
	// openSSEFinalize does.
	Instruction(&ir.Instruction{Opcode: "MOVB", Operands: []Op{AX, NewParamAddr("ret", 96)}})
	RET()
}

// avx512CopyBytes emits a runtime copy of `count` bytes, eight at a time then one at a time.
// It clobbers itr1, itr2 and t0, all of which are free outside Poly1305's own sequences.
func avx512CopyBytes(prefix string, from, to Mem, count Register) {
	MOVQ(count, itr1)
	XORQ(itr2, itr2)

	// Eight bytes at a time while a whole word remains, then the odd bytes. A byte-only loop
	// costs about 280 serialised moves on a 140-byte tail, which measured as a net regression
	// against the AVX2 kernel at WireGuard's 1420-byte payload.
	Label(prefix + "Copy8")
	CMPQ(itr1, Imm(8))
	JB(LabelRef(prefix + "Copy1"))
	src8, dst8 := from, to
	src8.Index, src8.Scale = itr2, 1
	dst8.Index, dst8.Scale = itr2, 1
	MOVQ(src8, t0)
	MOVQ(t0, dst8)
	ADDQ(Imm(8), itr2)
	SUBQ(Imm(8), itr1)
	JMP(LabelRef(prefix + "Copy8"))

	Label(prefix + "Copy1")
	TESTQ(itr1, itr1)
	JZ(LabelRef(prefix + "CopyDone"))
	src1, dst1 := from, to
	src1.Index, src1.Scale = itr2, 1
	dst1.Index, dst1.Scale = itr2, 1
	MOVB(src1, t0.As8())
	MOVB(t0.As8(), dst1)
	INCQ(itr2)
	DECQ(itr1)
	JMP(LabelRef(prefix + "Copy1"))

	Label(prefix + "CopyDone")
}

// avx512HashBuffer hashes ceil(count/16) sixteen-byte blocks out of the staging buffer. The
// buffer is zeroed beyond count, which is exactly the zero padding Poly1305 wants for a
// partial final block.
func avx512HashBuffer(prefix string, count Register) {
	MOVQ(count, itr1)
	XORQ(itr2, itr2)

	Label(prefix + "HashLoop")
	TESTQ(itr1, itr1)
	JLE(LabelRef(prefix + "HashDone"))
	block := avx512TailStore
	block.Index, block.Scale = itr2, 1
	polyAdd(block)
	polyMulBMI2()
	ADDQ(Imm(16), itr2)
	SUBQ(Imm(16), itr1)
	JMP(LabelRef(prefix + "HashLoop"))

	Label(prefix + "HashDone")
}

// avx512Tail handles a final partial group of inl bytes, where 0 < inl < 256, by staging it in
// a zeroed 256-byte buffer. This is what lets the kernel accept any length rather than only
// multiples of 256, so WireGuard's 1420 and 1440 and jumbo 8920 reach it and not just
// Tailscale's 1280.
//
// The two directions differ only in ordering. Seal hashes the ciphertext it just produced, so
// it must zero the staging buffer past inl first, because everything past inl is keystream
// rather than ciphertext and must not be absorbed. Open hashes the input ciphertext before
// decrypting, and its buffer is already zero-padded from the copy.
func avx512Tail(prefix string, initA, initB, initC, dRow VecVirtual, sealing bool) {
	Comment("Stage the partial final group in a zeroed 256-byte buffer")
	zero := ZMM()
	VPXORD(zero, zero, zero)
	for i := 0; i < avx512BlocksPerGroup; i++ {
		VMOVDQU32(zero, avx512TailStore.Offset(i*64))
	}
	// The byte count has to survive Poly1305, which clobbers t0 through t3 and RAX and RDX.
	// inl is the one register holding it that the poly emitters never touch, so it stays the
	// count and itr1 does the counting.
	avx512CopyBytes(prefix+"In", Mem{Base: inp}, avx512TailStore, inl)

	Comment("One group of keystream, xored over the whole staging buffer")
	work, init := avx512NewGroup(initA, initB, initC, dRow)
	for _, emit := range avx512RoundClosures(work) {
		emit()
	}
	avx512AddInit(work, init)
	blocks := avx512Transpose(work)

	if !sealing {
		/*
		   Open hashes the staged ciphertext here rather than before the rounds. It is correct
		   either side of them: the transpose works entirely in registers and the xor below is what
		   overwrites the staging buffer, so the ciphertext is still there at this point.

		   Before the rounds it had nothing to overlap with. avx512HashBuffer is a loop whose body
		   is a strictly serial Poly1305 multiply chain, and ahead of the rounds the only thing in
		   flight is the byte-wise copy it depends on. Here the chain overlaps the xor, the stores
		   and the copy-out. Measured in one binary against the previous ordering, byte-identical
		   apart from this move: 11 to 13ns per packet at 1420 bytes on Skylake-D, Ice Lake-SP,
		   Sapphire Rapids and Zen 5, and about 7 on Cascade Lake-W. At 1280, which is exactly five
		   groups and runs no tail at all, the two orderings measure the same to within 1ns, which
		   is what says the harness was measuring the change rather than noise.
		*/
		Comment("Open hashes the staged ciphertext, before the xor overwrites it")
		avx512HashBuffer(prefix, inl)
	}

	for i := 0; i < avx512BlocksPerGroup; i++ {
		VPXORD(avx512TailStore.Offset(i*64), blocks[i], blocks[i])
		VMOVDQU32(blocks[i], avx512TailStore.Offset(i*64))
	}

	if sealing {
		// Everything past the ciphertext is keystream now and must not be absorbed. Only the
		// bytes up to the next 16-byte boundary need clearing, because the hash reads
		// ceil(inl/16)*16 bytes and never looks beyond that: at most fifteen bytes, rather
		// than the up-to-255 a clear-to-end-of-buffer loop would touch.
		Comment("Zero from the end of the ciphertext to the next 16-byte boundary")
		MOVQ(inl, itr2)
		MOVQ(inl, itr1)
		ADDQ(Imm(15), itr1)
		ANDQ(I32(-16), itr1)
		Label(prefix + "ZeroPad")
		CMPQ(itr2, itr1)
		JAE(LabelRef(prefix + "ZeroPadDone"))
		pad := avx512TailStore
		pad.Index, pad.Scale = itr2, 1
		MOVB(Imm(0), pad)
		INCQ(itr2)
		JMP(LabelRef(prefix + "ZeroPad"))
		Label(prefix + "ZeroPadDone")

		Comment("Seal hashes the ciphertext it just produced")
		avx512HashBuffer(prefix, inl)
	}

	avx512CopyBytes(prefix+"Out", avx512TailStore, Mem{Base: oup}, inl)
	Comment("Advance the output cursor so the tag, or the tag compare, lands correctly")
	ADDQ(inl, oup)
	ADDQ(inl, inp)
}
