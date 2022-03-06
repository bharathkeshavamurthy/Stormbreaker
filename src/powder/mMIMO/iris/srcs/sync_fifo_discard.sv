`timescale 1ns / 1ps

`default_nettype none

module sync_fifo_discard # (
                            parameter integer FIFO_DEPTH       = 1024,   // Minimum required FIFO depth in number of words
                            parameter integer FIFO_AF_THRESH   =    8,   // AF asserted when less than AF_THRESH open FIFO "slots" remain
                            parameter integer FIFO_AE_THRESH   =    8,   // AE asserted when less than AE_THRESH words reside in the FIFO
                            parameter integer FIFO_WIDTH       =  128,   // FIFO IN/OUT WIDTH in bits
                            parameter integer ENABLE_DISCARD   =    1,   // Enable bulk discard operations? (1/0 = YES/NO)
                            parameter         RAM_TYPE         = "block" // Xilinx RAM implementation : "auto", "block", "distributed", or "ultra"
                           )
(
    input  wire                                     clk,
    input  wire                                     rst,

    output logic [(($clog2(FIFO_DEPTH) + 1) - 1):0] stat_count,
    output logic                                    stat_empty,
    output logic                                    stat_ae,
    output logic                                    stat_af,
    output logic                                    stat_full,

    input  wire  [              (FIFO_WIDTH - 1):0] in_dat,
    input  wire                                     in_dat_load,

    output logic [              (FIFO_WIDTH - 1):0] out_dat,
    output logic                                    out_dat_valid,

    input  wire                                     out_dat_unld,
    input  wire                                     out_dat_discard,
    input  wire  [(($clog2(FIFO_DEPTH) + 1) - 1):0] out_dat_discard_cnt,
    output logic                                    out_dat_discarding
);

localparam FF_DATA_WIDTH = FIFO_WIDTH;
localparam FF_DATA_MSB   = FF_DATA_WIDTH - 1;

localparam FF_ADDR_WIDTH = $clog2(FIFO_DEPTH); 
localparam FF_ADDR_MSB   = FF_ADDR_WIDTH - 1;

localparam FF_CNTR_WIDTH = FF_ADDR_WIDTH + 1;
localparam FF_CNTR_MSB   = FF_CNTR_WIDTH - 1;

localparam FF_MEM_DEPTH  = (2 ** FF_ADDR_WIDTH);
localparam FF_MEM_SIZE   = FF_MEM_DEPTH * FF_DATA_WIDTH;



// *****************************************************************************
// ** DISCARD CONTROL LOGIC ****************************************************
// *****************************************************************************

wire                  ctl_discard     = (ENABLE_DISCARD == 1) ? out_dat_discard                     : '0;
wire  [FF_CNTR_MSB:0] ctl_discard_cnt = (ENABLE_DISCARD == 1) ? out_dat_discard_cnt [FF_CNTR_MSB:0] : '0;
logic                 ctl_discarding;
logic [          1:0] ctl_discarding_cyc;

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        ctl_discarding           <= 1'b0;
        ctl_discarding_cyc [1:0] <= 2'b00;
    end

    else
    begin
        if (ENABLE_DISCARD == 1)
        begin
            ctl_discarding           <= ~ctl_discarding &  ctl_discard
                                      |  ctl_discarding & ~ctl_discarding_cyc [0];

            ctl_discarding_cyc [  0] <= ~ctl_discarding &  ctl_discard;
            ctl_discarding_cyc [  1] <=  1'b0;
        end

        else
        begin
            ctl_discarding           <= 1'b0;
            ctl_discarding_cyc [1:0] <= 2'b00;
        end
    end
end

assign out_dat_discarding = ctl_discarding;



// *****************************************************************************
// ** MEMORY INSTANTIATION *****************************************************
// *****************************************************************************

logic [FF_ADDR_MSB:0] mem_wt_addr;
wire  [FF_DATA_MSB:0] mem_wt_data           = in_dat [FF_DATA_MSB:0];
wire                  mem_wt_en             = in_dat_load;

logic [FF_CNTR_MSB:0] mem_rd_addr;
logic                 mem_rd_addr_nc;
wire  [FF_DATA_MSB:0] mem_rd_data;

logic [FF_CNTR_MSB:0] mem_rd_avail_cnt;
logic                 mem_rd_avail_cnt_nc;
wire                  mem_rd_avail          = mem_rd_avail_cnt [FF_CNTR_MSB];

(* KEEP = "TRUE" *) logic mem_rd_en;
(* KEEP = "TRUE" *) logic mem_rd_ce;
(* KEEP = "TRUE" *) logic mem_rd_clr;

xpm_memory_sdpram # (
                     .ADDR_WIDTH_A            ( FF_ADDR_WIDTH  ), // DECIMAL
                     .ADDR_WIDTH_B            ( FF_ADDR_WIDTH  ), // DECIMAL
                     .AUTO_SLEEP_TIME         ( 0              ), // DECIMAL
                     .BYTE_WRITE_WIDTH_A      ( FF_DATA_WIDTH  ), // DECIMAL
                     .CLOCKING_MODE           ( "common_clock" ), // String
                     .ECC_MODE                ( "no_ecc"       ), // String
                     .MEMORY_INIT_FILE        ( "none"         ), // String
                     .MEMORY_INIT_PARAM       ( "0"            ), // String
                     .MEMORY_OPTIMIZATION     ( "true"         ), // String
                     .MEMORY_PRIMITIVE        ( RAM_TYPE       ), // String
                     .MEMORY_SIZE             ( FF_MEM_SIZE    ), // DECIMAL
                     .MESSAGE_CONTROL         ( 0              ), // DECIMAL
                     .READ_DATA_WIDTH_B       ( FF_DATA_WIDTH  ), // DECIMAL
                     .READ_LATENCY_B          ( 2              ), // DECIMAL
                     .READ_RESET_VALUE_B      ( "0"            ), // String
                     .RST_MODE_A              ( "SYNC"         ), // String
                     .RST_MODE_B              ( "SYNC"         ), // String
                     .USE_EMBEDDED_CONSTRAINT ( 0              ), // DECIMAL
                     .USE_MEM_INIT            ( 0              ), // DECIMAL
                     .WAKEUP_TIME             ( "disable_sleep"), // String
                     .WRITE_DATA_WIDTH_A      ( FF_DATA_WIDTH  ), // DECIMAL
                     .WRITE_MODE_B            ( "read_first"   )  // String
                    )
    mem_inst (
              .clka           ( clk                        ),
              .addra          ( mem_wt_addr [FF_ADDR_MSB:0]),
              .dina           ( mem_wt_data [FF_DATA_MSB:0]),
              .ena            ( 1'b1                       ),
              .wea            ( mem_wt_en                  ),
              .clkb           ( clk                        ),
              .enb            ( mem_rd_en                  ),
              .regceb         ( mem_rd_ce                  ),
              .addrb          ( mem_rd_addr [FF_ADDR_MSB:0]),
              .doutb          ( mem_rd_data [FF_DATA_MSB:0]),
              .rstb           ( mem_rd_clr                 ),
              .injectdbiterra ( 1'b0                       ),
              .injectsbiterra ( 1'b0                       ),
              .dbiterrb       (                            ),
              .sbiterrb       (                            ),
              .sleep          ( 1'b0                       )
             );



// *****************************************************************************
// ** OUTPUT PIPELINE MANAGEMENT ***********************************************
// *****************************************************************************

logic       mem_ireg_valid;
logic       mem_oreg_valid;

wire        mem_ireg_load  = mem_rd_en;
wire        mem_ireg_unld  = mem_rd_ce;
wire        mem_oreg_load  = mem_rd_ce;
wire        mem_oreg_unld  = out_dat_unld;

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        mem_ireg_valid <= 1'b0;
        mem_oreg_valid <= 1'b0;
    end

    else
    begin
        mem_ireg_valid <= ~ctl_discard & ~ctl_discarding &  mem_ireg_load
                        | ~ctl_discard & ~ctl_discarding & ~mem_ireg_load & mem_ireg_valid & ~mem_ireg_unld;

        mem_oreg_valid <= ~ctl_discard & ~ctl_discarding &  mem_oreg_load
                        | ~ctl_discard & ~ctl_discarding & ~mem_oreg_load & mem_oreg_valid & ~mem_oreg_unld;
    end
end

always_comb
begin
    mem_rd_en  <= ~ctl_discard & ~ctl_discarding & mem_rd_avail & ~mem_ireg_valid
                | ~ctl_discard & ~ctl_discarding & mem_rd_avail &  mem_ireg_valid & ~mem_oreg_valid
                | ~ctl_discard & ~ctl_discarding & mem_rd_avail &  mem_ireg_valid &  mem_oreg_valid & mem_oreg_unld;

    mem_rd_ce  <= ~ctl_discard & ~ctl_discarding                &  mem_ireg_valid & ~mem_oreg_valid
                | ~ctl_discard & ~ctl_discarding                &  mem_ireg_valid &  mem_oreg_valid & mem_oreg_unld;

    mem_rd_clr <=  ctl_discard
                |                  ctl_discarding
                | ~ctl_discard  & ~ctl_discarding               & ~mem_ireg_valid & ~mem_oreg_valid
                | ~ctl_discard  & ~ctl_discarding               & ~mem_ireg_valid &  mem_oreg_valid & mem_oreg_unld;
end

assign out_dat       [FF_DATA_MSB:0] = mem_rd_data    [FF_DATA_MSB:0];
assign out_dat_valid                 = mem_oreg_valid;



// *****************************************************************************
// ** MEMORY ADDRESS AND AVAILABILITY LOGIC ************************************
// *****************************************************************************

logic [FF_CNTR_MSB:0] mem_rd_adjustment;
logic                 mem_rd_adjustment_nc;

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        mem_wt_addr          [FF_ADDR_MSB:0] <= '0;

        mem_rd_adjustment    [FF_CNTR_MSB:0] <= '0;
        mem_rd_adjustment_nc                 <= '0;

        mem_rd_addr          [FF_CNTR_MSB:0] <= '0;
        mem_rd_addr_nc                       <= '0;

        mem_rd_avail_cnt     [FF_CNTR_MSB:0] <= {1 << FF_CNTR_MSB} - 1;
        mem_rd_avail_cnt_nc                  <= '0;
    end

    else
    begin
        if (mem_wt_en) mem_wt_addr [FF_ADDR_MSB:0] <= mem_wt_addr [FF_ADDR_MSB:0] + 1;

        if (ctl_discard)
        begin
            {mem_rd_adjustment [FF_CNTR_MSB:0], mem_rd_adjustment_nc} <= {ctl_discard_cnt [FF_CNTR_MSB:0],           1'b0}
                                                                       - {             '0, mem_oreg_valid, mem_ireg_valid};
        end

        else
        begin
            {mem_rd_adjustment [FF_CNTR_MSB:0], mem_rd_adjustment_nc} <= '0;
        end

       {mem_rd_addr      [FF_CNTR_MSB:0], mem_rd_addr_nc     } <= {mem_rd_addr       [FF_CNTR_MSB:0],      1'b1}
                                                                + {mem_rd_adjustment [FF_CNTR_MSB:0], mem_rd_en};

       {mem_rd_avail_cnt [FF_CNTR_MSB:0], mem_rd_avail_cnt_nc} <= {mem_rd_avail_cnt  [FF_CNTR_MSB:0],      1'b0}
                                                                - {       {FF_CNTR_WIDTH{mem_wt_en}}, mem_rd_en}
                                                                - {mem_rd_adjustment [FF_CNTR_MSB:0],      1'b0};
    end
end



// *****************************************************************************
// ** FLAG GENERATION LOGIC ****************************************************
// *****************************************************************************

                    wire  ff_load = in_dat_load;
(* KEEP = "TRUE" *) logic ff_unld;

always_comb
begin
    ff_unld <= ~ctl_discard & ~ctl_discarding & out_dat_unld;
end

logic [FF_CNTR_MSB:0] ff_empty_cnt;
logic                 ff_empty_cnt_nc;

logic [FF_CNTR_MSB:0] ff_ae_cnt;
logic                 ff_ae_cnt_nc;

logic [FF_CNTR_MSB:0] ff_af_cnt;
logic                 ff_af_cnt_nc;

logic [FF_CNTR_MSB:0] ff_full_cnt;
logic                 ff_full_cnt_nc;

logic [FF_CNTR_MSB:0] ff_stat_adjustment;

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        ff_empty_cnt       [FF_CNTR_MSB:0] <= 0 - 1;
        ff_empty_cnt_nc                    <= '0;

        ff_ae_cnt          [FF_CNTR_MSB:0] <= 0 - FIFO_AE_THRESH;
        ff_ae_cnt_nc                       <= '0;

        ff_af_cnt          [FF_CNTR_MSB:0] <= {1 << FF_CNTR_WIDTH} - FIFO_AF_THRESH + 1;
        ff_af_cnt_nc                       <= '0;

        ff_full_cnt        [FF_CNTR_MSB:0] <= '0;
        ff_full_cnt_nc                     <= '0;

        ff_stat_adjustment [FF_CNTR_MSB:0] <= '0;
    end

    else
    begin
       {ff_empty_cnt [FF_CNTR_MSB:0], ff_empty_cnt_nc} <= {ff_empty_cnt       [FF_CNTR_MSB:0],    1'b0}
                                                        - {          {FF_CNTR_WIDTH{ff_load}}, ff_unld}
                                                        - {ff_stat_adjustment [FF_CNTR_MSB:0],    1'b0};

       {ff_ae_cnt    [FF_CNTR_MSB:0], ff_ae_cnt_nc   } <= {ff_ae_cnt          [FF_CNTR_MSB:0],    1'b0}
                                                        - {          {FF_CNTR_WIDTH{ff_load}}, ff_unld}
                                                        - {ff_stat_adjustment [FF_CNTR_MSB:0],    1'b0};

       {ff_af_cnt    [FF_CNTR_MSB:0], ff_af_cnt_nc   } <= {ff_af_cnt          [FF_CNTR_MSB:0],    1'b0}
                                                        - {          {FF_CNTR_WIDTH{ff_load}}, ff_unld}
                                                        - {ff_stat_adjustment [FF_CNTR_MSB:0],    1'b0};

       {ff_full_cnt  [FF_CNTR_MSB:0], ff_full_cnt_nc } <= {ff_full_cnt        [FF_CNTR_MSB:0],    1'b0}
                                                        - {          {FF_CNTR_WIDTH{ff_load}}, ff_unld}
                                                        - {ff_stat_adjustment [FF_CNTR_MSB:0],    1'b0};

        if   (ctl_discard) ff_stat_adjustment [FF_CNTR_MSB:0] <= ctl_discard_cnt [FF_CNTR_MSB:0];
        else               ff_stat_adjustment [FF_CNTR_MSB:0] <= '0;
    end
end

assign stat_count [FF_CNTR_MSB:0] = ff_full_cnt  [FF_CNTR_MSB:0];
assign stat_empty                 = ff_empty_cnt [  FF_CNTR_MSB];
assign stat_ae                    = ff_ae_cnt    [  FF_CNTR_MSB];
assign stat_af                    = ff_af_cnt    [  FF_CNTR_MSB];
assign stat_full                  = ff_full_cnt  [  FF_CNTR_MSB];

endmodule

`default_nettype wire
