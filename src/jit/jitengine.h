/* ES40 emulator -- JIT engine
 *
 * Per-CPU, direct-mapped cache of translation blocks. The dispatcher runs a
 * block's compiled safe-ALU prefix natively, then interprets the remainder.
 *
 * Blocks are keyed by VIRTUAL PC + ASN (like the icache), so the dispatch hot
 * path needs no address translation. A TB invalidation flushes the cache (see
 * flush()); a global (ASM) block matches any ASN.
 */
#if !defined(INCLUDED_JITENGINE_H)
#define INCLUDED_JITENGINE_H

#ifdef ES40_JIT

#include <cstdint>
#include "../config_debug.h"   // JIT_VERIFY

class CAlphaCPU;   // compiled blocks call back into the CPU for memory accesses

class CJitEngine
{
public:
  // 64K slots: the OS-active CPU's block working set (50K+) thrashed the old 16K direct-mapped
  // cache -- conflict evictions caused ~190K recompiles per 100M instructions on that CPU.
  static constexpr int      kCacheBits = 16;
  static constexpr int      kCacheEntries = 1 << kCacheBits;
  static constexpr uint64_t kIndexMask = (uint64_t) kCacheEntries - 1;

  // Reclaim executable memory once compiled code passes this many bytes, rather
  // than tearing down the asmjit runtime on every flush (see flush()).
  static constexpr uint64_t kReclaimBytes = 32 * 1024 * 1024;

  // Compiled block entry point. Runs the prefix on regs[0..31], calling back into
  // cpu for memory accesses; returns the number of instructions fully completed
  typedef uint32_t (*JitFn)(CAlphaCPU* cpu, uint64_t* regs);

  struct JitBlock
  {
    uint64_t tag;         // start VIRTUAL PC (validity tag / key)
    uint64_t phys;        // start physical PC (source bytes for compilation)
    uint32_t asn;         // address space number (key; ignored when asm_global)
    bool     asm_global;  // global (ASM) page: matches any ASN, like the icache
    uint32_t n_instr;     // instructions in the straight-line block
    bool     valid;
    JitFn    code;        // compiled safe-prefix, or null (prologue entry, for C calls)
    void*    jit_body;    // chained re-entry point (after the prologue); null when not compiled
    JitBlock* link;       // cached successor block (back-patched by the dispatcher); null = none
    uint32_t prefix_len;  // # safe ALU ops in code
    bool     compiled;    // compile has been attempted
    uint32_t body_off;    // jit_body's offset within code -- restores the chained entry on revalidate
    uint64_t src_sum;     // hash of the source words at compile time (revalidate vs self-mod)
    uint32_t hash_len;    // word count src_sum covers -- frozen at compile time; n_instr drifts
                          // (interrupt-truncated cold passes shrink it), so it must NOT key the hash
    uint64_t vgen;        // m_itb_gen + m_flush_gen at last full validation (phys + code bytes).
                          // Both counters are monotonic, so one sum compare detects either changing
                          // -- the single chain guard (see emit_chain / jit_indirect).
    uint64_t flush_gen;   // icache-flush generation at which the code bytes were last hash-validated;
                          // stale => lookup misses and revalidate_flushed() re-hashes (lazy IC_FLUSH)
  };

  // Byte offsets (from the CAlphaCPU*) of the fields the inline load fast path reads,
  // so compiled code can touch them via [rsi + offset]. Filled once by set_offsets().
  struct JitOffsets {
    uint32_t dpc_valid, dpc_virt_page, dpc_phys_base, dpc_cm, dpc_asn;  // offsets of slot [0][0]
    uint32_t dpc_stride, dpc_mask;   // direct-mapped page cache: per-slot byte stride, index mask
    uint32_t state_cm, state_asn0, dram_ptr, dram_size, state_pc;
    // For chaining: the budget ceiling and the interrupt-poll flags the compiled epilogue
    // checks before jumping on; link_from is where the epilogue records a link-patch request.
    uint32_t jit_budget, check_int, check_timers, link_from;
    uint32_t exc_addr, pal_base, sde;   // CALL_PAL: exc_addr save, PAL entry base, PALshadow enable
  };
  void set_offsets(const JitOffsets& o) { m_off = o; }

  explicit CJitEngine(int cpu_id = 0);   // cpu_id tags the stats/diagnostic prints
  ~CJitEngine();

  static inline uint64_t index_of(uint64_t virt_pc) { return (virt_pc >> 2) & kIndexMask; }

  // Virtual+ASN keyed: no translation on the dispatch hot path. A global (ASM)
  // block matches any ASN, mirroring the icache's hit rule. flush_gen-stale blocks
  // miss here; revalidate_flushed() resurrects them after a source-hash check.
  inline JitBlock* lookup(uint64_t virt_pc, uint32_t asn)
  {
    JitBlock& b = m_blocks[index_of(virt_pc)];
    return (b.valid && b.flush_gen == m_flush_gen && b.tag == virt_pc && (b.asm_global || b.asn == asn)) ? &b : nullptr;
  }

  // Lazy-flush survivor: hash-revalidate the slot in place (no interpreted pass, no re-record).
  JitBlock* revalidate_flushed(uint64_t virt_pc, uint32_t asn, uint64_t phys_pc, const uint8_t* dram);

  JitBlock* record(uint64_t virt_pc, uint64_t phys_pc, uint32_t asn, bool asm_global, uint32_t n_instr, const uint8_t* dram);
  void compile_block(JitBlock* b, const uint8_t* dram, uint64_t dram_size, void* read_helper, void* write_helper, void* opcdec_helper, void* hw_mfpr_helper, void* hw_ld_helper, void* hw_mtpr_helper, void* hw_st_helper, void* indirect_helper, void* read_locked_helper, void* stc_helper, void* misc_helper, void* read_vpte_helper, void* itof_helper, void* ftoi_helper, void* fltl_helper);
  void flush();
  void flush_non_global();   // flush only !asm_global blocks (the ASM-bit-clear / ASN icache flush)
  void reclaim_code();       // free ALL compiled code once past kReclaimBytes (cold-path only)

  // ITB-generation counter for the indirect-chain staleness check (jit_indirect). Bumped on every
  // I-stream TB invalidate (tbia/tbiap/tbis, ACCESS_EXEC) ... those can remap a code page WITHOUT
  // flushing the JIT, so a chained block could run stale bytes. 
  inline void     note_itb_invalidate() { ++m_itb_gen; }
  inline uint64_t vgen() const          { return m_itb_gen + m_flush_gen; }   // combined validation epoch

#ifdef JIT_VERIFY
  // Differential check: compiled result (jit) vs interpreter result (interp), r[0..30].
  void verify_compare(uint64_t blk_virt, const uint64_t* interp, const uint64_t* jit,
                      const uint32_t* words, uint32_t nwords);
#endif

#ifdef JIT_STATS
  // Accumulate native vs interpreted instruction counts; prints coverage periodically.
  void note_exec(uint32_t native_instr, uint32_t interp_instr);
#endif

private:
  JitBlock m_blocks[kCacheEntries];
  int      m_cpu_id;
  uint64_t m_recorded;
  uint64_t m_itb_gen = 0; // current ITB generation (bumped on every I-stream TB invalidate)
  uint64_t m_flush_gen = 0; // current icache-flush generation (bumped by flush(); lazy IC_FLUSH/IMB)
  uint64_t m_code_bytes;  // compiled bytes since last reclaim (see flush())
  void*    m_rt;          // asmjit::JitRuntime*
  JitOffsets m_off = {};  // field offsets for the inline load fast path
#ifdef JIT_VERIFY
  uint64_t m_v_exec, m_v_fail;
#endif
#ifdef JIT_STATS
  uint64_t m_stat_native, m_stat_interp;        // windowed: instrs run native vs interpreted
  uint64_t m_stat_hot, m_stat_miss;             // windowed: compiled-chain dispatches, interp blocks
  uint64_t m_stat_compiled, m_stat_plen_sum;    // cumulative: compiled blocks, sum of their lengths
  uint64_t m_term_op[64];                       // cumulative: opcode that ended a block's compiled prefix
  uint64_t m_pal_func[256];                     // cumulative: CALL_PAL function code that ended a block
  uint64_t m_mtpr_func[256];                    // cumulative: HW_MTPR (0x1d) IPR index that ended a block
  uint64_t m_hwld_func[16];                     // cumulative: HW_LD (0x1b) form (ins>>12 & 0xf) that ended a block
  bool     m_first_breaker_logged;              // one-shot guard for the punch-list print
#endif
};

#endif // ES40_JIT
#endif // INCLUDED_JITENGINE_H
