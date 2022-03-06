////////////////////////////////////////////////////////////////////////////////
//
// regs
//
////////////////////////////////////////////////////////////////////////////////
`ifndef __REG_SPACE
`define __REG_SPACE

/*
 * REG_INIT() instantiates internal register signals used for the macros below.
 * Place REG_INIT() before the always @ block in which use the macros below.
 */
`define REG_INIT(rst_in, addr_msk, addr_in, en_in, we_in, di_in, do_out, rdy_out)\
    wire                        __en_in   = en_in                 ;\
    wire                        __rst_in  = rst_in                ;\
    wire                        __we_in   = we_in                 ;\
    wire  [$bits(addr_in )-1:0] __addr_in = addr_in & addr_msk    ;\
    wire  [$bits(di_in   )-1:0] __di_in   = di_in                 ;\
    reg   [$bits(do_out  )-1:0] __do_out  = {$bits(do_out){1'b0}} ;\
    reg   [$bits(rdy_out )-1:0] __rdy_out = 1'b1                  ;\
    assign rdy_out = __rdy_out                                    ;\
    assign do_out  = __do_out

/*
 * Control the register bus ready signal.
 * The default behavior is to drive ready a cycle when enabled
 * which matches the register delay used by the register reads.
 * Use this macro once in the @ block (recommended).
 * or replace with some other logic for advanced use.
 */
`define REG_READY_DEFAULT                   \
    if(__en_in && !__we_in)                 \
        __do_out  <= {$bits(__do_out){1'b0}}; \
    __rdy_out <= __rst_in?1'b0:__en_in

/*
 * Declare a read-only register.
 * Use this within an always @ block.
 * Where rd_data must be declared as a register.
 */
`define REG_RO(addr, rd_data)                                                  \
    if(__en_in && !__we_in && (__addr_in == addr))                             \
        __do_out <= rd_data

/*!
 * Declare a write-only register
 * Use this within an always @ block.
 */
`define REG_WO(addr, wr_data, init)                                            \
    if(__rst_in)                                                               \
        wr_data <= init;                                                       \
    else if(__en_in &&  __we_in && (__addr_in == addr))                        \
        wr_data <= __di_in[$bits(wr_data)-1:0]
/*!
 * Declare a read and write register.
 * Use this within an always @ block.
 * Where rd_data must be declared as a register.
 */
`define REG_RW(addr, rd_data, wr_data, init)                                   \
    `REG_WO(addr, wr_data, init); else `REG_RO(addr, rd_data)

`define stb_rd 1
`define stb_wr 0

/*!
 * Strobe generation based on register bus reads and writes.
 * Where s_out is a register that will be strobed on read or write
 * based on the value of s_rw which can be stb_rd or stb_wr.
 * Use this within an always @ block.
 */
`define REG_STB(addr, s_type, s_out)                                           \
    if(__rdy_out && __en_in && (__we_in ^ s_type) &&  (__addr_in == addr))     \
        s_out <= 1'b1;                                                         \
    else                                                                       \
        s_out <= 1'b0

/*!
 * Strobe generation based on register bus reads and writes.
 * Where s_out is a register that will be strobed on read or write
 * based on the value of s_rw which can be stb_rd or stb_wr.
 * Use this within an always @* block
 */
`define REG_STB_D(addr, s_type, s_out)                                         \
    if(__rdy_out && __en_in && (__we_in ^ s_type) &&  (__addr_in == addr))     \
        s_out <= __di_in[$bits(s_out)-1:0];                                    \
    else                                                                       \
        s_out <= 'h0;

`endif //__REG_SPACE
