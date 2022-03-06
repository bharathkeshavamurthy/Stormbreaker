`timescale 1ns / 1ps

`default_nettype none

module stream_fifo_pkt_dropper # (
                                  parameter integer FIFO_DEPTH       = 1024,  // Minimum required DAT FIFO depth in number of transfers
                                  parameter integer FIFO_AF_THRESH   =    2,  // AF asserted when less than AF_THRESH data FIFO slots remain
                                  parameter integer FIFO_AE_THRESH   =    4,  // AE asserted when less than AE_THRESH data words reside in the FIFO

                                  parameter integer AUX_FF_DEPTH     =   32,  // Minimum required AUX FIFO depth in number of packets
                                  parameter integer AUX_FF_AF_THRESH =    2,  // AF asserted when less than AF_THRESH aux FIFO slots remain

                                  parameter integer MAX_PACKET_SIZE  = 1024,  // Maximum number of transfers (input TDATA words) in any packet
                                  parameter integer DATA_WIDTH       =  128,  // Width in bits of TDATA
                                  parameter integer USER_WIDTH       =   64,  // Width in bits of TUSER, if supported

                                  parameter integer HAS_TKEEP        =    0,  // Supports TKEEP signal?                   (1/0 = Y/N)
                                  parameter integer HAS_TUSER        =    0,  // Supports TUSER signal?                   (1/0 = Y/N)
                                  parameter integer STARTUP_WAIT     =    0,  // Wait for TLAST before first load?        (1/0 = Y/N)
                                 
                                  parameter integer KEEP_WIDTH       = (HAS_TKEEP == 1) ? ((DATA_WIDTH + 7) / 8) : 1
                                 )
(
    input  wire                                    clk,
    input  wire                                    rst,

    input  wire  [               (DATA_WIDTH-1):0] in_tdata,
    input  wire  [               (KEEP_WIDTH-1):0] in_tkeep,
    input  wire  [               (USER_WIDTH-1):0] in_tuser,
    input  wire                                    in_tlast,
    input  wire                                    in_tvalid,
    output logic                                   in_tready,

    input  wire                                    in_pkt_done,
    input  wire                                    in_pkt_discard,
    input  wire                                    in_pkt_disable,

    output logic [               (DATA_WIDTH-1):0] out_tdata,
    output logic [               (KEEP_WIDTH-1):0] out_tkeep,
    output logic [               (USER_WIDTH-1):0] out_tuser,
    output logic                                   out_tlast,
    output logic                                   out_tvalid,
    input  wire                                    out_tready,
    input  wire                                    out_discard_req,

    output logic                                   aux_pkt_started,
    output logic                                   aux_pkt_complete,
    output logic                                   aux_pkt_unloading,
    output logic                                   aux_pkt_discarding,
    output logic [($clog2(MAX_PACKET_SIZE)+1-1):0] aux_pkt_size,
    output logic                                   aux_pkt_haslast,
    output logic [               (KEEP_WIDTH-1):0] aux_pkt_tkeep,
    output logic                                   aux_pkt_discard,

    output wire                                    aux_dff_full,
    output wire                                    aux_dff_af,
    output wire                                    aux_dff_ae,

    output wire                                    aux_xff_full,
    output wire                                    aux_xff_af
);

`include "utils.sv"



// *****************************************************************************
// ** COMMONLY USED DERIVED PARAMETERS *****************************************
// *****************************************************************************

localparam DAT_FF_ADDR_WIDTH = $clog2(FIFO_DEPTH);
localparam DAT_FF_ADDR_MSB   = DAT_FF_ADDR_WIDTH - 1;

localparam DAT_FF_CNTR_WIDTH = $clog2(FIFO_DEPTH) + 1;
localparam DAT_FF_CNTR_MSB   = DAT_FF_CNTR_WIDTH - 1;

localparam DAT_FF_DATA_WIDTH = DATA_WIDTH + 1;
localparam DAT_FF_DATA_MSB   = DAT_FF_DATA_WIDTH - 1;

localparam IN_CYC_WIDTH      = $clog2(MAX_PACKET_SIZE);
localparam IN_CYC_MSB        = IN_CYC_WIDTH - 1;

localparam IN_CNT_WIDTH      = $clog2(MAX_PACKET_SIZE) + 1;
localparam IN_CNT_MSB        = IN_CNT_WIDTH - 1;

localparam AUX_FF_ADDR_WIDTH = $clog2(AUX_FF_DEPTH);
localparam AUX_FF_ADDR_MSB   = AUX_FF_ADDR_WIDTH - 1;

localparam AUX_FF_CNTR_WIDTH = $clog2(AUX_FF_DEPTH) + 1;
localparam AUX_FF_CNTR_MSB   = AUX_FF_CNTR_WIDTH - 1;

localparam AUX_FF_DATA_WIDTH = (HAS_TKEEP == 1) ? ($clog2(MAX_PACKET_SIZE) + 1 + KEEP_WIDTH + 2) : ($clog2(MAX_PACKET_SIZE) + 1 + 2);
localparam AUX_FF_DATA_MSB   = AUX_FF_DATA_WIDTH - 1;

localparam USR_FF_ADDR_WIDTH = $clog2(AUX_FF_DEPTH);
localparam USR_FF_ADDR_MSB   = USR_FF_ADDR_WIDTH - 1;

localparam USR_FF_CNTR_WIDTH = $clog2(AUX_FF_DEPTH) + 1;
localparam USR_FF_CNTR_MSB   = USR_FF_CNTR_WIDTH - 1;

localparam USR_FF_DATA_WIDTH = USER_WIDTH;
localparam USR_FF_DATA_MSB   = USR_FF_DATA_WIDTH - 1;



// *****************************************************************************
// ** INPUT SEQUENCING AND CONTROL *********************************************
// *****************************************************************************

logic [IN_CYC_MSB:0] in_cyc;        // Position within current packet (0 through ?)
logic                in_cyc_first;  // Asserted for first packet position
logic                in_cyc_last;   // Asserted for last  packet position

(* KEEP = "TRUE" *) logic in_cyc_en;
(* KEEP = "TRUE" *) logic in_cyc_wrap;

logic                in_pkt_enabled;
logic                in_pkt_started;

(* KEEP = "TRUE" *) logic in_pkt_error;
(* KEEP = "TRUE" *) logic in_pkt_valid;
(* KEEP = "TRUE" *) logic in_pkt_start;
(* KEEP = "TRUE" *) logic in_pkt_end;

always_comb
begin
    casex ({in_pkt_enabled, in_pkt_disable, in_tready, in_tvalid, in_cyc_first, in_cyc_last, in_tlast})

        7'b0X_XX_XXX,          // INPUT DISABLED
        7'b10_0X_XXX,          // INPUT ENABLED (AND NOT BEING DISABLED), TREADY = 0
        7'b10_X0_XXX :         // INPUT ENABLED (AND NOT BEING DISABLED), TVALID = 0
                       begin
                           in_cyc_en    <= 1'b0;
                           in_cyc_wrap  <= 1'b0;
                           in_pkt_error <= 1'b0;
                           in_pkt_valid <= 1'b0;
                           in_pkt_start <= 1'b0;
                           in_pkt_end   <= 1'b0;
                       end
        7'b10_11_100 :         // VALID START OF PACKET
                       begin
                           in_cyc_en    <= 1'b1;
                           in_cyc_wrap  <= 1'b0;
                           in_pkt_error <= 1'b0;
                           in_pkt_valid <= 1'b1;
                           in_pkt_start <= 1'b1;
                           in_pkt_end   <= 1'b0;
                       end
        7'b10_11_000 :         // VALID CONTINUATION OF PACKET
                       begin
                           in_cyc_en    <= 1'b1;
                           in_cyc_wrap  <= 1'b0;
                           in_pkt_error <= 1'b0;
                           in_pkt_valid <= 1'b1;
                           in_pkt_start <= 1'b0;
                           in_pkt_end   <= 1'b0;
                       end
        7'b10_11_001,
        7'b10_11_011 :         // VALID END OF PACKET
                       begin
                           in_cyc_en    <= 1'b1;
                           in_cyc_wrap  <= 1'b1;
                           in_pkt_error <= 1'b0;
                           in_pkt_valid <= 1'b1;
                           in_pkt_start <= 1'b0;
                           in_pkt_end   <= 1'b1;
                       end
        7'b10_11_11X,          // SEQUENCING ERROR (CYC_FIRST WITH    CYC_LAST)
        7'b10_11_1X1,          // SEQUENCING ERROR (CYC_FIRST WITH    TLAST   )
        7'b10_11_X10 :         // SEQUENCING ERROR (CYC_LAST  WITHOUT TLAST   )
                       begin
                           in_cyc_en    <= 1'b1;
                           in_cyc_wrap  <= 1'b1;
                           in_pkt_error <= in_pkt_started;
                           in_pkt_valid <= 1'b0;
                           in_pkt_start <= 1'b0;
                           in_pkt_end   <= in_pkt_started;
                       end
        7'b11_XX_XXX :         // INPUT ENABLED, BUT BEING DISABLED
                       begin
                           in_cyc_en    <= 1'b1;
                           in_cyc_wrap  <= 1'b1;
                           in_pkt_error <= in_pkt_started;
                           in_pkt_valid <= 1'b0;
                           in_pkt_start <= 1'b0;
                           in_pkt_end   <= in_pkt_started;
                       end
    endcase
end

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        in_cyc       [IN_CYC_MSB:0] <= '0;
        in_cyc_first                <= 1'b1;
        in_cyc_last                 <= 1'b0;
    end

    else if (in_cyc_en)
    begin
        if   (in_cyc_wrap) in_cyc [IN_CYC_MSB:0] <= '0;
        else               in_cyc [IN_CYC_MSB:0] <= in_cyc [IN_CYC_MSB:0] + 1; 

        in_cyc_first <=  in_cyc_wrap;
        in_cyc_last  <= ~in_cyc_wrap & (in_cyc [IN_CYC_MSB:0] == (MAX_PACKET_SIZE - 2));
    end
end

logic [4:0] in_pkt_error_recov;

always_ff @ (posedge clk)
begin
    if      (rst)
    begin
        in_pkt_error_recov [4:0] <= 5'b10000;
        in_pkt_enabled           <= 1'b0;
        in_pkt_started           <= 1'b0;
    end

    else
    begin
        if (in_pkt_disable | in_pkt_error)
        begin
            if   (in_pkt_disable) in_pkt_error_recov [4:0] <= 5'b10000;
            else                  in_pkt_error_recov [4:0] <= 5'b00000;

            in_pkt_enabled <= 1'b0;
            in_pkt_started <= 1'b0;
        end

        else
        begin
            in_pkt_error_recov [4:0] <=  in_pkt_error_recov [4:0] + {4'h0, ~in_pkt_error_recov [4]};

            in_pkt_enabled <= ~in_pkt_enabled & (STARTUP_WAIT == 0) & in_pkt_error_recov [4]
                            | ~in_pkt_enabled & (STARTUP_WAIT == 1) & in_pkt_error_recov [4] & in_tready &  in_tvalid &  in_tlast
                            |  in_pkt_enabled;

            in_pkt_started <= ~in_pkt_started &  in_pkt_start
                            |  in_pkt_started & ~in_pkt_end;
        end
    end
end



// *****************************************************************************
// ** INPUT ISOLATION REGISTERS ************************************************
// *****************************************************************************

logic [(DATA_WIDTH - 1):0] ireg_tdata;
logic                      ireg_tlast;
logic [(KEEP_WIDTH - 1):0] ireg_tkeep;
logic [(USER_WIDTH - 1):0] ireg_tuser;
logic [      IN_CNT_MSB:0] ireg_size;

logic                      ireg_data_valid;
logic                      ireg_user_valid;
logic                      ireg_size_valid;

logic                      ireg_discard;

always_ff @ (posedge clk)
begin
    ireg_tdata   [(DATA_WIDTH - 1):0] <= in_tdata [(DATA_WIDTH - 1):0];
    ireg_tlast                        <= in_tlast;
    ireg_tkeep   [(KEEP_WIDTH - 1):0] <= (HAS_TKEEP == 1) ? in_tkeep [(KEEP_WIDTH - 1):0] : '0;
    ireg_tuser   [(USER_WIDTH - 1):0] <= (HAS_TUSER == 1) ? in_tuser [(USER_WIDTH - 1):0] : '0;
    ireg_size    [      IN_CNT_MSB:0] <= {1'b0, in_cyc [IN_CYC_MSB:0]} + {'0, in_pkt_valid};

    if (rst)
    begin
        ireg_data_valid <= 1'b0;
        ireg_user_valid <= 1'b0;
        ireg_size_valid <= 1'b0;
    end

    else
    begin
        ireg_data_valid <= in_pkt_valid;
        ireg_user_valid <= in_pkt_start;
        ireg_size_valid <= in_pkt_end;
    end
end

logic                load_aux_a;
logic                load_aux_b;
logic                unld_aux_a;
logic                unld_aux_b;
logic                aux_a_valid;
logic                aux_b_valid;

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        load_aux_a   <= 1'b0;
        load_aux_b   <= 1'b0;
        unld_aux_a   <= 1'b0;
        unld_aux_b   <= 1'b0;
        aux_a_valid  <= 1'b0;
        aux_b_valid  <= 1'b0;
        ireg_discard <= 1'b0;
    end

    else
    begin
        load_aux_a   <=  in_pkt_end     &  in_pkt_valid;
        load_aux_b   <=  in_pkt_end     & ~in_pkt_valid;

        unld_aux_a   <=  in_pkt_done                       // Normal packet completion mechanism
                      |  in_pkt_end     & ~in_pkt_valid    // Next packet ends with error
                      |  in_pkt_enabled &  in_pkt_disable; // Input disabled during next packet

        unld_aux_b   <=  load_aux_b;

        aux_a_valid  <= ~aux_a_valid    & load_aux_a & ~unld_aux_a
                      |  aux_a_valid                 & ~unld_aux_a;

        aux_b_valid  <= ~aux_b_valid    & load_aux_b & ~unld_aux_b
                      |  aux_b_valid                 & ~unld_aux_b;

        ireg_discard <=  in_pkt_done    &  in_pkt_discard
                      |  in_pkt_end     & ~in_pkt_valid
                      |  in_pkt_enabled &  in_pkt_disable;
    end
end

logic [     IN_CNT_MSB:0] aux_a_size;
logic [ (KEEP_WIDTH-1):0] aux_a_tkeep;
logic [     IN_CNT_MSB:0] aux_b_size;
logic [ (KEEP_WIDTH-1):0] aux_b_tkeep;

logic [AUX_FF_DATA_MSB:0] aux_data;
logic                     aux_data_valid;

always_ff @ (posedge clk)
begin
    if (load_aux_a)
    begin
        aux_a_size  [    IN_CNT_MSB:0] <= ireg_size  [IN_CNT_MSB:0];
        aux_a_tkeep [(KEEP_WIDTH-1):0] <= (HAS_TKEEP == 1) ? ireg_tkeep [(KEEP_WIDTH-1):0] : '0;
    end

    if (load_aux_b)
    begin
        aux_b_size  [    IN_CNT_MSB:0] <= ireg_size  [IN_CNT_MSB:0];
        aux_b_tkeep [(KEEP_WIDTH-1):0] <= (HAS_TKEEP == 1) ? ireg_tkeep [(KEEP_WIDTH-1):0] : '0;
    end

    //                                                              {                          KEEP,      DISCARD, HASLAST,               PACKET SIZE}
    casex ({unld_aux_a, load_aux_a, unld_aux_b})
        3'bXX1 : aux_data [AUX_FF_DATA_MSB:0] <= (HAS_TKEEP == 1) ? {aux_b_tkeep [(KEEP_WIDTH-1):0],         1'b1,    1'b0, aux_b_size [IN_CNT_MSB:0]} : {        1'b1, 1'b0, aux_b_size [IN_CNT_MSB:0]};
        3'b110 : aux_data [AUX_FF_DATA_MSB:0] <= (HAS_TKEEP == 1) ? {ireg_tkeep  [(KEEP_WIDTH-1):0], ireg_discard,    1'b1, ireg_size  [IN_CNT_MSB:0]} : {ireg_discard, 1'b1, ireg_size  [IN_CNT_MSB:0]};
        3'b100 : aux_data [AUX_FF_DATA_MSB:0] <= (HAS_TKEEP == 1) ? {aux_a_tkeep [(KEEP_WIDTH-1):0], ireg_discard,    1'b1, aux_a_size [IN_CNT_MSB:0]} : {ireg_discard, 1'b1, aux_a_size [IN_CNT_MSB:0]};
        3'b0X0 : aux_data [AUX_FF_DATA_MSB:0] <= '0;
    endcase

    if   (rst) aux_data_valid <= 1'b0;
    else       aux_data_valid <=  unld_aux_b              &  aux_b_valid
                               | ~unld_aux_b & unld_aux_a &  aux_a_valid
                               | ~unld_aux_b & unld_aux_a & ~aux_a_valid & load_aux_a;
end



// *****************************************************************************
// ** DATA FIFO ****************************************************************
// *****************************************************************************

localparam DAT_FF_MEM_DEPTH = 2 ** DAT_FF_ADDR_WIDTH;

wire  [  DAT_FF_DATA_MSB:0] dff_in_data = {ireg_tlast, ireg_tdata [(DATA_WIDTH - 1):0]};
wire                        dff_in_load =  ireg_data_valid;

wire  [  DAT_FF_DATA_MSB:0] dff_out_data;
wire                        dff_out_valid;

(* KEEP = "TRUE" *) logic   dff_out_unld;

logic                       dff_out_discard;
logic [  DAT_FF_CNTR_MSB:0] dff_out_discard_cnt;
wire                        dff_out_discarding;

sync_fifo_discard # (
                     .FIFO_DEPTH     (DAT_FF_MEM_DEPTH      ),
                     .FIFO_AF_THRESH (2                     ),
                     .FIFO_AE_THRESH (2                     ),
                     .FIFO_WIDTH     (DAT_FF_DATA_WIDTH     ),
                     .ENABLE_DISCARD (1                     ),
                     .RAM_TYPE       ("block"               )
                    )
    dff_inst (
              .clk                 (clk                                     ),
              .rst                 (rst                                     ),
              .stat_count          (                                        ),
              .stat_empty          (                                        ),
              .stat_ae             (                                        ),
              .stat_af             (                                        ),
              .stat_full           (                                        ),
              .in_dat              ( dff_in_data         [DAT_FF_DATA_MSB:0]),
              .in_dat_load         ( dff_in_load                            ),
              .out_dat             ( dff_out_data        [DAT_FF_DATA_MSB:0]),
              .out_dat_valid       ( dff_out_valid                          ),
              .out_dat_unld        ( dff_out_unld                           ),
              .out_dat_discard     ( dff_out_discard                        ),
              .out_dat_discard_cnt ( dff_out_discard_cnt [DAT_FF_CNTR_MSB:0]),
              .out_dat_discarding  ( dff_out_discarding                     )
             );

wire [(DATA_WIDTH-1):0] dff_out_tdata = dff_out_data [0 +: DATA_WIDTH];
wire                    dff_out_tlast = dff_out_data [     DATA_WIDTH];



// *****************************************************************************
// ** AUX FIFO *****************************************************************
// *****************************************************************************

localparam AUX_FF_MEM_DEPTH = 2 ** AUX_FF_ADDR_WIDTH;

wire  [  AUX_FF_DATA_MSB:0] xff_in_data = aux_data       [AUX_FF_DATA_MSB:0];
wire                        xff_in_load = aux_data_valid;

wire  [  AUX_FF_DATA_MSB:0] xff_out_data;
wire                        xff_out_valid;

(* KEEP = "TRUE" *) logic   xff_out_unld;

logic                       xff_out_discard;
logic [  AUX_FF_CNTR_MSB:0] xff_out_discard_cnt;
wire                        xff_out_discarding;

sync_fifo_discard # (
                     .FIFO_DEPTH     (AUX_FF_MEM_DEPTH      ),
                     .FIFO_AF_THRESH (2                     ), 
                     .FIFO_AE_THRESH (2                     ), 
                     .FIFO_WIDTH     (AUX_FF_DATA_WIDTH     ), 
                     .ENABLE_DISCARD (1                     ), 
                     .RAM_TYPE       ("auto"                )
                    )
    xff_inst (
              .clk                 (clk                                     ),
              .rst                 (rst                                     ),
              .stat_count          (                                        ),
              .stat_empty          (                                        ),
              .stat_ae             (                                        ),
              .stat_af             (                                        ),
              .stat_full           (                                        ),
              .in_dat              ( xff_in_data         [AUX_FF_DATA_MSB:0]),
              .in_dat_load         ( xff_in_load                            ),
              .out_dat             ( xff_out_data        [AUX_FF_DATA_MSB:0]),
              .out_dat_valid       ( xff_out_valid                          ),
              .out_dat_unld        ( xff_out_unld                           ),
              .out_dat_discard     ( xff_out_discard                        ),
              .out_dat_discard_cnt ( xff_out_discard_cnt [AUX_FF_CNTR_MSB:0]),
              .out_dat_discarding  ( xff_out_discarding                     )
             );

wire [    IN_CNT_MSB:0] xff_out_size        = xff_out_data [(IN_CNT_MSB  ):0];
wire                    xff_out_haslast     = xff_out_data [(IN_CNT_MSB+1)  ];
wire                    xff_out_discard_req = xff_out_data [(IN_CNT_MSB+2)  ];
wire [(KEEP_WIDTH-1):0] xff_out_tkeep       = (HAS_TKEEP == 1) ? xff_out_data [(IN_CNT_MSB+3) +: KEEP_WIDTH] : '0;



// *****************************************************************************
// ** USER FIFO ****************************************************************
// *****************************************************************************

localparam USR_FF_MEM_DEPTH = 2 ** USR_FF_ADDR_WIDTH;

wire  [USR_FF_DATA_MSB:0] uff_in_data = (HAS_TUSER == 1) ? ireg_tuser      [(USER_WIDTH-1):0] : '0;
wire                      uff_in_load = (HAS_TUSER == 1) ? ireg_user_valid                    : 1'b0;

wire  [USR_FF_DATA_MSB:0] uff_out_data;
wire                      uff_out_valid;

(* KEEP = "TRUE" *) logic uff_out_unld;

logic                     uff_out_discard;
logic [USR_FF_CNTR_MSB:0] uff_out_discard_cnt;
wire                      uff_out_discarding;

sync_fifo_discard # (
                     .FIFO_DEPTH     (USR_FF_MEM_DEPTH      ),
                     .FIFO_AF_THRESH (2                     ), 
                     .FIFO_AE_THRESH (2                     ), 
                     .FIFO_WIDTH     (USR_FF_DATA_WIDTH     ), 
                     .ENABLE_DISCARD (1                     ), 
                     .RAM_TYPE       ("auto"                )
                    )
    uff_inst (
              .clk                 (clk                                     ),
              .rst                 (rst                                     ),
              .stat_count          (                                        ),
              .stat_empty          (                                        ),
              .stat_ae             (                                        ),
              .stat_af             (                                        ),
              .stat_full           (                                        ),
              .in_dat              ( uff_in_data         [USR_FF_DATA_MSB:0]),
              .in_dat_load         ( uff_in_load                            ),
              .out_dat             ( uff_out_data        [USR_FF_DATA_MSB:0]),
              .out_dat_valid       ( uff_out_valid                          ),
              .out_dat_unld        ( uff_out_unld                           ),
              .out_dat_discard     ( uff_out_discard                        ),
              .out_dat_discard_cnt ( uff_out_discard_cnt [USR_FF_CNTR_MSB:0]),
              .out_dat_discarding  ( uff_out_discarding                     )
             );



// *****************************************************************************
// ** OUTPUT SEQUENCING AND CONTROL ********************************************
// *****************************************************************************

(* KEEP = "TRUE" *) logic                    out_unload_start;
(* KEEP = "TRUE" *) logic                    out_discard_start;

(* KEEP = "TRUE" *) logic                    out_tvalid_int;
(* KEEP = "TRUE" *) logic [(KEEP_WIDTH-1):0] out_tkeep_int;

(* KEEP = "TRUE" *) logic                    aux_pkt_started_int;
(* KEEP = "TRUE" *) logic                    aux_pkt_complete_int;
(* KEEP = "TRUE" *) logic                    aux_pkt_discard_int;

logic       out_unloading;
logic       out_discarding;
logic [1:0] out_discard_cyc;

assign dff_out_discard = out_discard_cyc [0];
assign xff_out_discard = out_discard_cyc [0];
assign uff_out_discard = out_discard_cyc [0];

always_comb
begin
     aux_pkt_started_int  <=                    ~out_discarding & ~out_unloading & ~xff_out_valid & dff_out_valid
                           |                    ~out_discarding & ~out_unloading &  xff_out_valid
                           |                                       out_unloading &  xff_out_valid;

     aux_pkt_complete_int <=                    ~out_discarding & ~out_unloading &  xff_out_valid
                           |                                       out_unloading &  xff_out_valid;

     aux_pkt_discard_int  <=                    ~out_discarding & ~out_unloading &  xff_out_valid                 & ( xff_out_discard_req | ~xff_out_haslast)
                           |                                       out_unloading &  xff_out_valid                 & ( xff_out_discard_req | ~xff_out_haslast);

     out_tvalid_int       <=                    ~out_discarding & ~out_unloading &  xff_out_valid & dff_out_valid & (~xff_out_discard_req &  xff_out_haslast)
                           |                                       out_unloading &  xff_out_valid;

     out_discard_start    <=  out_discard_req & ~out_discarding & ~out_unloading &  xff_out_valid;

     out_unload_start     <= ~out_discard_req & ~out_discarding & ~out_unloading &  xff_out_valid & dff_out_valid & (~xff_out_discard_req &  xff_out_haslast) &  out_tready;

     dff_out_unld         <= ~out_discard_req & ~out_discarding & ~out_unloading &  xff_out_valid & dff_out_valid & (~xff_out_discard_req &  xff_out_haslast) &  out_tready
                           |                                       out_unloading &  xff_out_valid                                                             &  out_tready;

     xff_out_unld         <= ~out_discard_req & ~out_discarding & ~out_unloading &  xff_out_valid & dff_out_valid & (~xff_out_discard_req &  xff_out_haslast) &  out_tready & out_tlast
                           |                                       out_unloading &  xff_out_valid                                                             &  out_tready & out_tlast;

     uff_out_unld         <= ~out_discard_req & ~out_discarding & ~out_unloading &  xff_out_valid & dff_out_valid & (~xff_out_discard_req &  xff_out_haslast) &  out_tready & out_tlast
                           |                                       out_unloading &  xff_out_valid                                                             &  out_tready & out_tlast;

/*
    dff_out_unld         <= ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid & ~xff_out_discard_req & out_tready
                          |  out_unloading                                      &  xff_out_valid                        & out_tready;

    xff_out_unld         <= ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid & ~xff_out_discard_req & out_tready & out_tlast
                          |  out_unloading                                      &  xff_out_valid                        & out_tready & out_tlast;

    uff_out_unld         <= ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid & ~xff_out_discard_req & out_tready & out_tlast
                          |  out_unloading                                      &  xff_out_valid                        & out_tready & out_tlast;

    out_unload_start     <= ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid & ~xff_out_discard_req & out_tready;

    out_discard_start    <= ~out_unloading & ~out_discarding &  out_discard_req &  xff_out_valid;

    out_tvalid_int       <= ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid & ~xff_out_discard_req
                          |  out_unloading                                      &  xff_out_valid;

    aux_pkt_started_int  <= ~out_unloading & ~out_discarding & ~out_discard_req & ~xff_out_valid &  dff_out_valid
                          | ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid &  dff_out_valid
                          |  out_unloading                                      &  xff_out_valid &  1'b0;

    aux_pkt_complete_int <= ~out_unloading & ~out_discarding & ~out_discard_req &  xff_out_valid
                          |  out_unloading                                      &  xff_out_valid &  1'b0;

    aux_pkt_discard_int  <= ~out_unloading & ~out_discarding                    &  xff_out_valid &  xff_out_discard_req;
*/
    casex ({xff_out_valid, xff_out_haslast, dff_out_tlast})
        4'b0XX : out_tkeep_int [(KEEP_WIDTH-1):0] <= (HAS_TKEEP == 1) ? '0                               : '0;
        4'b10X,
        4'b1X0 : out_tkeep_int [(KEEP_WIDTH-1):0] <= (HAS_TKEEP == 1) ? {KEEP_WIDTH{1'b1}}               : '0;
        4'b111 : out_tkeep_int [(KEEP_WIDTH-1):0] <= (HAS_TKEEP == 1) ? xff_out_tkeep [(KEEP_WIDTH-1):0] : '0;
     endcase
end

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        out_unloading         <= '0;
        out_discarding        <= '0;
        out_discard_cyc [1:0] <= '0;

        dff_out_discard_cnt [DAT_FF_CNTR_MSB:0] <= '0;
        xff_out_discard_cnt [AUX_FF_CNTR_MSB:0] <= '0;
        uff_out_discard_cnt [USR_FF_CNTR_MSB:0] <= '0;
    end

    else
    begin
        out_unloading       <= out_unload_start
                             | out_unloading     & (~out_tready | ~out_tvalid | ~out_tlast);

        out_discarding      <= out_discard_start 
                             | out_discarding    & ~out_discard_cyc [1];

        out_discard_cyc [0] <= out_discard_start;
        out_discard_cyc [1] <= out_discard_cyc [0];

        if (out_discard_start)
        begin
            dff_out_discard_cnt [DAT_FF_CNTR_MSB:0] <= {'0, xff_out_data [IN_CNT_MSB:0]};
            xff_out_discard_cnt [AUX_FF_CNTR_MSB:0] <= {'0, 1'b1};
            uff_out_discard_cnt [USR_FF_CNTR_MSB:0] <= {'0, 1'b1};
        end
        
        else
        begin
            dff_out_discard_cnt [DAT_FF_CNTR_MSB:0] <= '0;
            xff_out_discard_cnt [AUX_FF_CNTR_MSB:0] <= '0;
            uff_out_discard_cnt [USR_FF_CNTR_MSB:0] <= '0;
        end

    end
end



// *****************************************************************************
// ** MODULE OUTPUTS ***********************************************************
// *****************************************************************************

assign out_tdata        [(DATA_WIDTH-1):0] = dff_out_tdata  [(DATA_WIDTH-1):0];
assign out_tlast                           = dff_out_tlast;
assign aux_pkt_size     [   IN_CNT_MSB:0]  = xff_out_size   [    IN_CNT_MSB:0];
assign aux_pkt_haslast                     = xff_out_haslast;

assign aux_pkt_tkeep    [(KEEP_WIDTH-1):0] = (HAS_TKEEP == 1) ? xff_out_tkeep [(KEEP_WIDTH-1):0] : '0;
assign out_tuser        [(USER_WIDTH-1):0] = (HAS_TUSER == 1) ? uff_out_data  [(USER_WIDTH-1):0] : '0;
assign out_tkeep        [(KEEP_WIDTH-1):0] = (HAS_TKEEP == 1) ? out_tkeep_int [(KEEP_WIDTH-1):0] : '0;

assign out_tvalid                          = out_tvalid_int;
assign aux_pkt_started                     = aux_pkt_started_int;
assign aux_pkt_complete                    = aux_pkt_complete_int;
assign aux_pkt_unloading                   = out_unloading;
assign aux_pkt_discarding                  = out_discarding;
assign aux_pkt_discard                     = aux_pkt_discard_int;



// *****************************************************************************
// ** EXTERNAL FULL AND ALMOST FULL FLAG GENERATION ****************************
// *****************************************************************************

logic [DAT_FF_CNTR_MSB:0] dat_ff_full_cnt;
logic                     dat_ff_full_cnt_nc;

logic [DAT_FF_CNTR_MSB:0] dat_ff_af_cnt;
logic                     dat_ff_af_cnt_nc;

logic [DAT_FF_CNTR_MSB:0] dat_ff_ae_cnt;
logic                     dat_ff_ae_cnt_nc;

logic [AUX_FF_CNTR_MSB:0] aux_ff_full_cnt;
logic [              1:0] aux_ff_full_cnt_nc;

logic [AUX_FF_CNTR_MSB:0] aux_ff_af_cnt;
logic [              1:0] aux_ff_af_cnt_nc;

always_ff @ (posedge clk)
begin
    if (rst)
    begin
        dat_ff_full_cnt [DAT_FF_CNTR_MSB:0] <= {1 << DAT_FF_CNTR_MSB} - DAT_FF_MEM_DEPTH;

        dat_ff_af_cnt   [DAT_FF_CNTR_MSB:0] <= {1 << DAT_FF_CNTR_MSB} - DAT_FF_MEM_DEPTH + FIFO_AF_THRESH   - 1;

        dat_ff_ae_cnt   [DAT_FF_CNTR_MSB:0] <= 0 - FIFO_AE_THRESH;

        aux_ff_full_cnt [AUX_FF_CNTR_MSB:0] <= {1 << AUX_FF_CNTR_MSB} - AUX_FF_MEM_DEPTH;

        aux_ff_af_cnt   [AUX_FF_CNTR_MSB:0] <= {1 << AUX_FF_CNTR_MSB} - AUX_FF_MEM_DEPTH + AUX_FF_AF_THRESH - 1;

        dat_ff_full_cnt_nc       <= 1'b0;
        dat_ff_af_cnt_nc         <= 1'b0;
        dat_ff_ae_cnt_nc         <= 1'b0;
        aux_ff_full_cnt_nc [1:0] <= 2'b00;
        aux_ff_af_cnt_nc   [1:0] <= 2'b00;
    end

    else
    begin
       {dat_ff_full_cnt [DAT_FF_CNTR_MSB:0], dat_ff_full_cnt_nc} <= {    dat_ff_full_cnt [DAT_FF_CNTR_MSB:0],         1'b0}
                                                                  - {      {DAT_FF_CNTR_WIDTH{in_pkt_valid}}, dff_out_unld}
                                                                  - {dff_out_discard_cnt [DAT_FF_CNTR_MSB:0],         1'b0};

       {dat_ff_af_cnt   [DAT_FF_CNTR_MSB:0], dat_ff_af_cnt_nc  } <= {      dat_ff_af_cnt [DAT_FF_CNTR_MSB:0],         1'b0}
                                                                  - {      {DAT_FF_CNTR_WIDTH{in_pkt_valid}}, dff_out_unld}
                                                                  - {dff_out_discard_cnt [DAT_FF_CNTR_MSB:0],         1'b0};

       {dat_ff_ae_cnt   [DAT_FF_CNTR_MSB:0], dat_ff_ae_cnt_nc  } <= {      dat_ff_ae_cnt [DAT_FF_CNTR_MSB:0],         1'b0}
                                                                  - {      {DAT_FF_CNTR_WIDTH{in_pkt_valid}}, dff_out_unld}
                                                                  - {dff_out_discard_cnt [DAT_FF_CNTR_MSB:0],         1'b0};

       {aux_ff_full_cnt [AUX_FF_CNTR_MSB:0], aux_ff_full_cnt_nc [1:0]} <= {aux_ff_full_cnt [AUX_FF_CNTR_MSB:0],                        2'b00}
                                                                        - {    {AUX_FF_CNTR_WIDTH{in_pkt_end}}, xff_out_unld, xff_out_discard};

       {aux_ff_af_cnt   [AUX_FF_CNTR_MSB:0], aux_ff_af_cnt_nc   [1:0]} <= {  aux_ff_af_cnt [AUX_FF_CNTR_MSB:0],                        2'b00}
                                                                        - {    {AUX_FF_CNTR_WIDTH{in_pkt_end}}, xff_out_unld, xff_out_discard};
    end
end

(* KEEP = "TRUE" *) logic in_tready_int;

always_comb
begin
    in_tready_int <=  in_pkt_enabled & ~dat_ff_full_cnt [DAT_FF_CNTR_MSB] & ~aux_ff_af_cnt [AUX_FF_CNTR_MSB]
                   | ~in_pkt_enabled;
end

assign in_tready   =  in_tready_int;

assign aux_dff_full =  dat_ff_full_cnt [DAT_FF_CNTR_MSB];
assign aux_dff_af   =  dat_ff_af_cnt   [DAT_FF_CNTR_MSB];
assign aux_dff_ae   =  dat_ff_ae_cnt   [DAT_FF_CNTR_MSB];

assign aux_xff_full =  aux_ff_full_cnt [AUX_FF_CNTR_MSB];
assign aux_xff_af   =  aux_ff_af_cnt   [AUX_FF_CNTR_MSB];

endmodule

`default_nettype wire
