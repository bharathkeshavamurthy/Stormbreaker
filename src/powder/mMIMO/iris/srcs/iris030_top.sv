//----------------------------------------------------------------------------------
//-- Iris top level module
//--
//-- Contains processor core, axi stream flows and router, RF and GPIO ties-ins...
//--  * Processor tie in with ABP registers and AXI DMA for linux network stack access.
//--  * Router core for stream routing between GTX/Aurora cores, linux, and RF streams.
//--  * LMS7 data IQ line to source synchronous busses for ADC and DAC interfaces.
//--  * RF core interface for framed streaming between adaption for ADC/DAC interfaces.
//--  * IO management for misc control GPIOs, RF module control, and ICNT bus comms.
//----------------------------------------------------------------------------------

`default_nettype none

module iris030_top # (
    parameter VERSION = 64'h8000_0000,

    parameter SYS_CLK_PERIOD_PS = 10000, //100 MHz

    parameter UDP_SERVICE_PORT = 55132,

    //the fifo depth @128 bit for RF transmit
    parameter RF_TX_FIFO_DEPTH = 32*1024,

    //support up to 8k jumbo frames for the tx buffering
    parameter TX_BUFF_MTU_PKT = 512)
(
//#################### @BEGIN GEN PINS@ ####################
    input  wire [3:0] brd_rev,
    inout  wire       pmod_1,
    inout  wire       pmod_2,
    inout  wire       pmod_3,
    inout  wire       pmod_4,
    inout  wire       pmod_7,

    //LMS connections
    inout  wire       lms_dig_rst,           //LMS digital reset
    inout  wire       lms_hw_rxen,           //RX hard power off
    inout  wire       lms_hw_txen,           //TX hard power off
    inout  wire       clkbuff_rst_n,         //Si5344 reset line (low = reset]
    inout  wire       clkbuff_lock,          //Si5344 loss of lock indicator (high = no lock]
    inout  wire       clkbuff_irq,           //Si5344 status change interrupt (high = change, must clear]
    inout  wire       fpga_red_led,          //rev B was active low
    output wire       fpga_grn_led_n,
    inout  wire [16:1] rfmod_gpio,           //various GPIO lines to Iris-FE modules
    inout  wire       lms_spi_sclk,          //Serial port clock, positive edge sensitive, CMOS
    inout  wire       lms_spi_sdio,          //Serial port data in/out, CMOS
    inout  wire       lms_spi_sdo,           //Serial port data out, CMOS
    inout  wire       lms_spi_sen,           //Serial enable
    output wire       lms_diq1_iqseldir,     //IQSEL direction control for port 1.
    input  wire       lms_diq1_mclk,         //input source synchronous clock for port 1.
    output wire       lms_diq1_fclk,         //output feedback clock for port 1.
    output wire       lms_diq1_txnrx,        //IQ flag in RXTXIQ mode enable flag in JESD207 mode for port 1.
    output wire [11:0] lms_diq1_d,           //data bus for port 1
    input  wire       lms_diq2_iqseldir,     //IQSEL direction control for port 2.
    input  wire       lms_diq2_mclk,         //input source synchronous clock for port 2.
    output wire       lms_diq2_fclk,         //output feedback clock for port 2.
    output wire       lms_diq2_txnrx,        //IQ flag in RXTXIQ mode enable flag in JESD207 mode for port 2.
    input  wire [11:0] lms_diq2_d,           //data bus for port 2

    //silabs gt reference
    input  wire       mgtrefclk0_p,
    input  wire       mgtrefclk0_n,

    //backplane connections
    output wire [3:0] mgttx_p,
    output wire [3:0] mgttx_n,
    input  wire [3:0] mgtrx_p,
    input  wire [3:0] mgtrx_n,

    //uplink feedback clock
    output wire       icnt_up_aux_clk_n,
    output wire       icnt_up_aux_clk_p,

    //uplink IO
    input  wire       icnt_up_psnt_n_in,
    inout  wire       icnt_up_sclk,
    inout  wire       icnt_up_is_head_n,
    output wire       icnt_up_psnt_n_out,
    inout  wire       icnt_up_sda,

    //dnlink IOs
    input  wire       icnt_dn_psnt_n_in,
    inout  wire       icnt_dn_sda,
    output wire       icnt_dn_psnt_n_out,
    inout  wire       icnt_dn_sclk,
    inout  wire       icnt_dn_trigger,
    inout  wire       icnt_dn_aux,
    inout  wire       icnt_dn_is_tail_n,

//#################### @END GEN PINS@   ####################

    // fixed IO and DDR signals for the processor
    inout wire [ 14 : 0 ] DDR_addr,
    inout wire [  2 : 0 ] DDR_ba,
    inout wire            DDR_cas_n,
    inout wire            DDR_ck_n,
    inout wire            DDR_ck_p,
    inout wire            DDR_cke,
    inout wire            DDR_cs_n,
    inout wire [  3 : 0 ] DDR_dm,
    inout wire [ 31 : 0 ] DDR_dq,
    inout wire [  3 : 0 ] DDR_dqs_n,
    inout wire [  3 : 0 ] DDR_dqs_p,
    inout wire            DDR_odt,
    inout wire            DDR_ras_n,
    inout wire            DDR_reset_n,
    inout wire            DDR_we_n,
    inout wire            FIXED_IO_ddr_vrn,
    inout wire            FIXED_IO_ddr_vrp,
    inout wire [ 53 : 0 ] FIXED_IO_mio,
    inout wire            FIXED_IO_ps_clk,
    inout wire            FIXED_IO_ps_porb,
    inout wire            FIXED_IO_ps_srstb
);

    `include "utils.sv"

    //main system clock
    wire SYS_rst;
    (* DONT_TOUCH = "TRUE" *) wire SYS_clk;

    ////////////////////////////////////////////////////////////////////
    //brd_rev straps - sample for all of time (pullups required), pulldown
    //stuffed to modify (inverting logic)
    //BRD_REV[3..0] =~0b0000 = Revision B
    //BRD_REV[3..0] =~0b0001 = Revision C
    ////////////////////////////////////////////////////////////////////
    reg [3:0] rev_r;
    always @(posedge SYS_clk)
        if(SYS_rst)
            rev_r <= ~brd_rev;

    wire rev_b = (rev_r == 4'b0000);
    wire rev_c = (rev_r == 4'b0001) || (rev_r == 4'b0010); //rev C and E

    //APB register interface
    wire [ 31 : 0 ] APB_M_paddr   [4:1];
    wire            APB_M_penable [4:1];
    wire [ 31 : 0 ] APB_M_prdata  [4:1];
    wire            APB_M_pready  [4:1];
    wire            APB_M_psel    [4:1];
    wire            APB_M_pslverr [4:1];
    wire [ 31 : 0 ] APB_M_pwdata  [4:1];
    wire            APB_M_pwrite  [4:1];

    //test mode controls
    reg blink_enable = 1'b0;
    reg [31:0] high_thermal_shutdown_event_kwd = 32'b0;
    reg high_thermal_shutdown_event_accepted  = 1'b0;

    //misc gpio controls
    reg [9:0] gpio_reg;
    wire [9:0] gpio_in;
    reg gpio_write;
    reg fe_pgood;
    wire lms_tdd_enb;
    wire lms_hw_rstn;

    //gt controls/status
    reg  [31:0] sys_clk_cycles;
    wire [31:0] upl_user_clk_cycles;
    wire [31:0] dnl_user_clk_cycles;
    wire uplink_active, dnlink_active;
    wire [31:0] gtx_clk_counter;
    wire [31:0] SYS_link_status;
    reg  [31:0] SYS_link_control;
    reg  [31:0] SYS_gt_ctrl_up;
    reg  [31:0] SYS_gt_ctrl_dn;
    reg SYS_gt_ctrl_up_stb;
    reg SYS_gt_ctrl_dn_stb;
    wire [31:0] rx_error_counter_up;
    wire [31:0] rx_error_counter_dn;
    reg  [7:0] aurora_link_events_up;
    reg  [7:0] aurora_link_events_dn;
    reg  [7:0] clock_events;
    reg  [15:0] dsp_pkt_drops;

    //triggers
    wire internal_trigger;
    wire SYS_trigger;

    //ethernet controls
    reg [47:0] local_hw_addr;
    reg GEM_DMA_ready;
    reg NET_DMA_ready;
    reg GT_DNL_squelch;
    reg GT_UPL_squelch;

    //data data wires (upper 16 I, lower 16 Q)
    wire DATA_clk ;
    wire DATA_rst ;
    wire [31 : 0] RX_data_a ;
    wire [31 : 0] RX_data_b ;
    wire TX_active;
    wire [31 : 0] TX_data_a;
    wire [31 : 0] TX_data_b;
    wire SYS_tx_active;
    wire RFC_trigger;
    wire SYS_RFC_trigger;

`ifdef CORR
    wire [ 31 : 0 ] coe_paddr   ;
    wire            coe_penable ;
    wire [ 31 : 0 ] coe_prdata  ;
    wire            coe_pready  ;
    wire            coe_psel    ;
    wire            coe_pslverr ;
    wire [ 31 : 0 ] coe_pwdata  ;
    wire            coe_pwrite  ;

//    wire [ 11 : 0 ] faros_corr_dbg_addr;
//    wire            faros_corr_dbg_clk;
//    wire [ 31 : 0 ] faros_corr_dbg_din;
//    wire [ 31 : 0 ] faros_corr_dbg_dout;
//    wire            faros_corr_dbg_en;
//    wire            faros_corr_dbg_rst;
//    wire [ 3 : 0 ]  faros_corr_dbg_we;
//
//    wire [ 11 : 0 ] faros_corr_in_dbg_addr;
//    wire            faros_corr_in_dbg_clk;
//    wire [ 31 : 0 ] faros_corr_in_dbg_din;
//    wire [ 31 : 0 ] faros_corr_in_dbg_dout;
//    wire            faros_corr_in_dbg_en;
//    wire            faros_corr_in_dbg_rst;
//    wire [ 3 : 0 ]  faros_corr_in_dbg_we;

    //argos interfaces;
    wire FAROS_fbclk ;
    wire FAROS_rst ;
    wire SYSGEN_clk ;
    wire SYSGEN_rst ;
    wire SYSGEN_dcm_locked ;
    wire [31 : 0] sysgen_clk_counter;
    wire [31 : 0] FAROS_rx_data_a;
    wire [31 : 0] FAROS_rx_data_b;
    wire [31 : 0] sync_counter;
    wire FAROS_trigger;
    reg RFC_rst;
    reg [9:0] corr_test_addr;
    wire [31:0] corr_test_data;

    // faros corr
    wire [31 : 0] faros_corr_0_i;
    wire [31 : 0] faros_corr_0_q;
    wire [31 : 0] faros_corr_0_delayed_i;
    wire [31 : 0] faros_corr_0_delayed_q;
    wire [23 : 0] faros_corr_iq_in;
    wire [23 : 0] faros_corr_iq_in_delayed;
    reg [31 : 0] faros_corr_en;
    reg [31 : 0] threshold_scale;
    reg [31 : 0] threshold_scale_1x;
`endif

    //axi stream interfaces
    wire [127:0]    AXIS_DSP_TX_tdata_buf;
    wire [15 :0]    AXIS_DSP_TX_tkeep_buf;
    wire            AXIS_DSP_TX_tlast_buf;
    wire            AXIS_DSP_TX_tvalid_buf;
    wire            AXIS_DSP_TX_tready_buf;

    wire [127 : 0]  AXIS_UPL_RX_tdata  ;
    wire            AXIS_UPL_RX_tlast  ;
    wire            AXIS_UPL_RX_tready ;
    wire            AXIS_UPL_RX_tvalid ;
    wire [127 : 0]  AXIS_UPL_TX_tdata  ;
    wire            AXIS_UPL_TX_tlast  ;
    wire            AXIS_UPL_TX_tready ;
    wire            AXIS_UPL_TX_tvalid ;

    wire [127 : 0]  AXIS_DNL_RX_tdata  ;
    wire            AXIS_DNL_RX_tlast  ;
    wire            AXIS_DNL_RX_tready ;
    wire            AXIS_DNL_RX_tvalid ;
    wire [127 : 0]  AXIS_DNL_TX_tdata  ;
    wire            AXIS_DNL_TX_tlast  ;
    wire            AXIS_DNL_TX_tready ;
    wire            AXIS_DNL_TX_tvalid ;

    wire [127 : 0]  AXIS_DSP_RX_tdata  ;
    wire [0   : 0]  AXIS_DSP_RX_tuser  ;
    wire [15  : 0]  AXIS_DSP_RX_tkeep  ;
    wire            AXIS_DSP_RX_tlast  ;
    wire            AXIS_DSP_RX_tready ;
    wire            AXIS_DSP_RX_tvalid ;
    wire [127 : 0]  AXIS_DSP_TX_tdata  ;
    wire [15  : 0]  AXIS_DSP_TX_tkeep  ;
    wire            AXIS_DSP_TX_tlast  ;
    wire            AXIS_DSP_TX_tready ;
    wire            AXIS_DSP_TX_tvalid ;
    wire            AXIS_DSP_TX_almost_full ;

    wire [127 : 0 ] AXIS_NET_S2MM_tdata  ;
    wire            AXIS_NET_S2MM_tlast  ;
    wire            AXIS_NET_S2MM_tready ;
    wire            AXIS_NET_S2MM_tvalid ;
    wire [127 : 0 ] AXIS_NET_MM2S_tdata  ;
    wire            AXIS_NET_MM2S_tlast  ;
    wire            AXIS_NET_MM2S_tready ;
    wire            AXIS_NET_MM2S_tvalid ;

    wire [127 : 0 ] AXIS_GEM_S2MM_tdata  ;
    wire            AXIS_GEM_S2MM_tlast  ;
    wire            AXIS_GEM_S2MM_tready ;
    wire            AXIS_GEM_S2MM_tvalid ;
    wire [127 : 0 ] AXIS_GEM_MM2S_tdata  ;
    wire            AXIS_GEM_MM2S_tlast  ;
    wire            AXIS_GEM_MM2S_tready ;
    wire            AXIS_GEM_MM2S_tvalid ;

    wire [127:0]    AXIS_RBF_RX_tdata;
    wire            AXIS_RBF_RX_tlast;
    wire            AXIS_RBF_RX_tvalid;
    wire            AXIS_RBF_RX_tready;
    wire [127:0]    AXIS_RBF_TX_tdata;
    wire            AXIS_RBF_TX_tlast;
    wire            AXIS_RBF_TX_tvalid;
    wire            AXIS_RBF_TX_tready;
    wire            AXIS_RBF_TX_almost_full;

    // fifo debugging
    reg [31:0] AXIS_RBF_RX_pkts;
    reg [31:0] AXIS_RBF_TX_pkts;
    reg [31:0] AXIS_DSP_RX_pkts;
    reg [31:0] AXIS_DSP_TX_pkts;
    reg [31:0] AXIS_UPL_RX_pkts;
    reg [31:0] AXIS_UPL_TX_pkts;
    reg [31:0] AXIS_DNL_RX_pkts;
    reg [31:0] AXIS_DNL_TX_pkts;
    reg [31:0] AXIS_NET_S2MM_pkts;
    reg [31:0] AXIS_NET_MM2S_pkts;

    `EVENT_COUNTER(SYS_clk, SYS_rst, sys_clk_cycles, 1'b1);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_RBF_RX_pkts, AXIS_RBF_RX_tlast && AXIS_RBF_RX_tvalid && AXIS_RBF_RX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_RBF_TX_pkts, AXIS_RBF_TX_tlast && AXIS_RBF_TX_tvalid && AXIS_RBF_TX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_DSP_RX_pkts, AXIS_DSP_RX_tlast && AXIS_DSP_RX_tvalid && AXIS_DSP_RX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_DSP_TX_pkts, AXIS_DSP_TX_tlast && AXIS_DSP_TX_tvalid && AXIS_DSP_TX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_UPL_RX_pkts, AXIS_UPL_RX_tlast && AXIS_UPL_RX_tvalid && AXIS_UPL_RX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_UPL_TX_pkts, AXIS_UPL_TX_tlast && AXIS_UPL_TX_tvalid && AXIS_UPL_TX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_DNL_RX_pkts, AXIS_DNL_RX_tlast && AXIS_DNL_RX_tvalid && AXIS_DNL_RX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_DNL_TX_pkts, AXIS_DNL_TX_tlast && AXIS_DNL_TX_tvalid && AXIS_DNL_TX_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_NET_S2MM_pkts, AXIS_NET_S2MM_tlast && AXIS_NET_S2MM_tvalid && AXIS_NET_S2MM_tready);
    `EVENT_COUNTER(SYS_clk, SYS_rst, AXIS_NET_MM2S_pkts, AXIS_NET_MM2S_tlast && AXIS_NET_MM2S_tvalid && AXIS_NET_MM2S_tready);

    //sync manager and trigger generator
    wire [31:0] SYS_msg_sync_status;
    wire SYS_sync_done, SYS_trigger_req_int, SYS_req_shutdown, SYS_req_restart;
    reg SYS_send_ping_up, SYS_send_ping_dn;
    reg SYS_decr_pulse_in, SYS_incr_pulse_in;
    reg SYS_sync_delays, SYS_trigger_gen;
    wire [63:0] SYS_time64;

    //virtual uart
    wire [15:0] uart_rx_msg_tdata;
    wire uart_rx_msg_tvalid;
    reg  uart_rx_msg_tready;
    reg  [15:0] uart_tx_msg_tdata;
    reg  uart_tx_msg_tvalid;
    wire uart_tx_msg_tready;

    //LMS DIQ TRX controls
    reg trxiq_config_write;
    reg [31:0] trxiq_config_data;
    reg [31:0] trx_test_data_a_tx, trx_test_data_b_tx;
    wire [31:0] trx_test_data_a_rx, trx_test_data_b_rx;
    wire [31:0] data_clk_counter;
    //PRBS controls
    reg [15:0] trx_prbs_ctrl;
    wire [63:0] trx_prbs_e;
    wire [31:0] trx_prbs_stat;

    //i2c tie in for interconnect peripherals
    wire I2C0_SCL_I, I2C0_SCL_O, I2C0_SCL_T;
    wire I2C0_SDA_I, I2C0_SDA_O, I2C0_SDA_T;

    //test control regs
    reg  [31:0] icnt_test_ctrl ;
    wire [31:0] icnt_test_stat;
    wire [31:0] icnt_test_em;

    //lms7 spi state machine
    reg [31:0] lms7_spi_wr_data;
    reg lms7_spi_wr_valid, lms7_spi_rd_ready;
    wire [31:0] lms7_spi_rd_data;
    wire lms7_spi_rd_valid, lms7_spi_wr_ready;
    reg lms7_prog_spi_wr_valid;
    reg [15:0] lms7_prog_spi_wr_addr;
    reg [31:0] lms7_prog_spi_wr_data;
    reg [24:0] lms7_spi_pattern;

    //gain controls for lms7
    reg tx_gain_ctrl_en;
    reg tx_gain_ctrl_mode; // 1: control from FPGA, 0: control from SW
    reg [1:0] tx_gain_step;
    reg [5:0] tx_gain_start;
    reg [5:0] tx_gain_stop;
    reg [4:0] tx_gain_pad_a;
    reg [4:0] tx_gain_pad_b;
    reg [2:0] rsvd_a, rsvd_b;
    wire [4:0] cur_gain_pad;
    wire [4:0] gain_pad [0:1];
    assign gain_pad[0] = tx_gain_ctrl_mode ? cur_gain_pad : tx_gain_pad_a;
    assign gain_pad[1] = tx_gain_ctrl_mode ? cur_gain_pad : tx_gain_pad_b;
    wire rx_gain_ctrl_en;
    wire [3:0] gain_lna [0:1];
    wire [1:0] gain_tia [0:1];
    wire gain_override = tx_gain_ctrl_en | rx_gain_ctrl_en;
    reg [31:0] gain_read = {gain_override, 18'b0, gain_pad[1], 3'b0, gain_pad[0]};
    wire [11:0] gain_pga [0:1];

    wire [3:0] gain_lna1 [0:1];
    wire [15:0] gain_lna2 [0:1];
    wire [15:0] gain_attn [0:1];

    // AGC
    wire agc_trigger;
    reg agc_en;
    reg agc_reset;
    reg [31:0] iq_thresh;
    reg [7:0] num_samps_sat;
    reg [7:0] max_num_samps_agc;
    reg [31:0] rssi_target;
    reg [7:0] wait_count_thresh;
    reg [7:0] small_jump;
    reg [7:0] big_jump;
    reg [3:0] cbrs_rev;
    reg [2:0] num_sat_stages_in;
    reg [2:0] num_fine_stages_in;

    // AGC Debugging
    reg test_gain_settings;
    reg [3:0] gain_lna_in;
    reg [1:0] gain_tia_in;
    reg [11:0] gain_pga_in;
    reg [3:0] gain_lna1_in;
    reg [15:0] gain_lna2_in;
    reg [15:0] gain_attn_in;
    reg [7:0] init_gain_in;

    // RSSI
    reg [31:0] meas_rssi;
    wire [15:0] meas_rssi_a;
    wire [15:0] meas_rssi_b;

//AGC
    wire agc_on;
    wire [11:0] gain_pga_out  [0:1];
    wire [ 3:0] gain_lna_out  [0:1];
    wire [ 1:0] gain_tia_out  [0:1];
    wire        gain_lna1_out [0:1];
    wire        gain_lna2_out [0:1];
    wire [ 2:1] gain_attn1_out [0:1];
    wire [ 2:1] gain_attn2_out [0:1];
    reg agc_table_wr_stb;
    wire agc_table_wr_en;
    wire [ 6:0] agc_table_addr;
    wire [23:0] agc_table_data;
    wire [31:0] agc_table_read_data;
    reg  [31:0] agc_ctrl;
    reg         agc_ctrl_stb;
    wire [31:0] agc_status;
    wire [ 7:0] agc_debug;
    reg  [31:0] agc_table_cmd_data;


    // Packet Detection
    reg [31:0] pkt_det_thresh;
    reg [15:0] pkt_det_num_samps;
    reg pkt_det_en;
    reg new_frame_flag;

    wire [3:0] irqs0 = {SYS_req_restart, SYS_req_shutdown};

    wire [3:0] build_variant;
    `ifdef ENABLE_RRH
    assign build_variant = 4'b0010;
    `elsif CORR
    assign build_variant = 4'b0100;
    `else
    assign build_variant = 4'b0001;
    `endif

    ////////////////////////////////////////////////////////////////////
    // Clock domain change for AGC/pktDet/RSSI
    ////////////////////////////////////////////////////////////////////
    // ==== IN ====
    //                     228 =           3+                    3+                16+              32+             8+            16+           16+           4+           12+          2+            4+         4+
    //                                  227:225               224:222            221:206          205:174        173:166        165:150       149:134       133:130      129:118      117:116      115:112    111:108
    reg [227:0] agc_reg_tick_sys = {num_fine_stages_in, num_sat_stages_in, pkt_det_num_samps, pkt_det_thresh, init_gain_in, gain_attn_in, gain_lna2_in, gain_lna1_in, gain_pga_in, gain_tia_in, gain_lna_in, cbrs_rev,
    //                             8+          8+            8+             32+              8+               8+            32+           1+            1+             1+            1
    //                           107:100     99:92          91:84          83:52            51:44            43:36          35:4          3             2              1             0
                                big_jump, small_jump, wait_count_thresh, rssi_target, max_num_samps_agc, num_samps_sat, iq_thresh, new_frame_flag, pkt_det_en, test_gain_settings, agc_en};
    reg [227:0] agc_reg_tick_data;
    xpm_cdc_array_single # (4,0,0,1,0,228) agc_ctrlIN_tick(SYS_clk, agc_reg_tick_sys, DATA_clk, agc_reg_tick_data);

    // ==== OUT ====
    //                         87 =            1+            4+          2+           12+           4+          16+           16 +          32
    //                                         86           85:82       81:80        79:68         67:64       63:48          47:32         31:0
    reg [86:0] agc_reg_tick_out_data = {rx_gain_ctrl_en, gain_lna[0], gain_tia[0], gain_pga[0], gain_lna1[0], gain_lna2[0], gain_attn[0], meas_rssi};
    reg [86:0] agc_reg_tick_out_sys;
    xpm_cdc_array_single # (4,0,0,1,0,87) agc_ctrlOUT_tick(DATA_clk, agc_reg_tick_out_data, SYS_clk, agc_reg_tick_out_sys);

    ////////////////////////////////////////////////////////////////////
    // Top level registers
    ////////////////////////////////////////////////////////////////////
    `include "system_regs.vh"

    `REG_INIT(SYS_rst, 32'hffff, APB_M_paddr[1], APB_M_psel[1] & APB_M_penable[1],
             APB_M_pwrite[1], APB_M_pwdata[1], APB_M_prdata[1], APB_M_pready[1]);

    always @(posedge SYS_clk) begin
        `REG_READY_DEFAULT;

        //module configuration
        `REG_RO(0, VERSION[31:0]);
        `REG_RO(4, VERSION[63:32]);

        //special test mode bits
        `REG_WO(4, blink_enable, 1'b0);

        //readback configuration parameters so software can match the FPGA
        `REG_RO(8, UDP_SERVICE_PORT);
        `REG_RO(12, RF_TX_FIFO_DEPTH);

        //user revision straps
        `REG_RO(16, {build_variant, fe_pgood, rev_r});

        //networking and link management
        `REG_WO(16, GEM_DMA_ready, 1'b0);
        `REG_WO(20, NET_DMA_ready, 1'b0);
        `REG_WO(24, local_hw_addr[31:0], '0);
        `REG_WO(28, local_hw_addr[47:32], '0);
        `REG_RW(32, SYS_link_status, SYS_link_control, '1);
        `REG_WO(36, {GT_UPL_squelch, GT_DNL_squelch}, 2'b0);

        //gpio register
        `REG_RW(40, {irqs0, gpio_in}, gpio_reg, '0);
        `REG_STB(40, `stb_wr, gpio_write);

        //triggers
        `REG_RO(44, SYS_msg_sync_status);
        `REG_STB_D(44, `stb_wr, {
            SYS_send_ping_up, SYS_send_ping_dn,
            SYS_decr_pulse_in, SYS_incr_pulse_in,
            SYS_sync_delays, SYS_trigger_gen});

        //trxiq config
        `REG_WO(48, trxiq_config_data, 32'b0);
        `REG_STB(48, `stb_wr, trxiq_config_write);
        `REG_RW(52, trx_test_data_a_rx, trx_test_data_a_tx, 32'b0);
        `REG_RW(56, trx_test_data_b_rx, trx_test_data_b_tx, 32'b0);
`ifdef CORR
        `REG_WO(60, faros_corr_en, 32'b0);
        `REG_WO(64, RFC_rst, 1'b0);
`endif
        `REG_STB_D(68, `stb_wr, high_thermal_shutdown_event_kwd); 
        //link debugging
        `REG_RO(68, gtx_clk_counter);
        `REG_RO(72, data_clk_counter);
`ifdef CORR
        `REG_RO(84, {SYSGEN_dcm_locked, sysgen_clk_counter[30:0]});
`endif
        `REG_RW(88, gain_read, {rsvd_b, tx_gain_pad_b, rsvd_a, tx_gain_pad_a, tx_gain_ctrl_en, tx_gain_ctrl_mode, tx_gain_step, tx_gain_stop, tx_gain_start}, 32'h1f1f0000);
`ifdef CORR
        `REG_WO(92, threshold_scale, 32'b0);
        `REG_RO(92, sync_counter);
`endif

        `REG_RW(96, SYS_gt_ctrl_up, SYS_gt_ctrl_up, 32'h80000400);
        `REG_RW(100, SYS_gt_ctrl_dn, SYS_gt_ctrl_dn, 32'h80000400);
        `REG_STB(96, `stb_wr, SYS_gt_ctrl_up_stb);
        `REG_STB(100, `stb_wr, SYS_gt_ctrl_dn_stb);
        `REG_RO(104, upl_user_clk_cycles);
        `REG_RO(108, dnl_user_clk_cycles);
        `REG_RO(112, sys_clk_cycles);

        //fifo debugging
        `REG_RO(120, clock_events);
        `REG_RO(124, dsp_pkt_drops);
        `REG_RO(128, AXIS_RBF_RX_pkts);
        `REG_RO(132, AXIS_RBF_TX_pkts);
        `REG_RO(136, AXIS_DSP_RX_pkts);
        `REG_RO(140, AXIS_DSP_TX_pkts);
        `REG_RO(144, AXIS_UPL_RX_pkts);
        `REG_RO(148, AXIS_UPL_TX_pkts);
        `REG_RO(152, AXIS_DNL_RX_pkts);
        `REG_RO(156, AXIS_DNL_TX_pkts);
        `REG_RO(160, AXIS_NET_S2MM_pkts);
        `REG_RO(164, AXIS_NET_MM2S_pkts);
        `REG_RO(168, rx_error_counter_up);
        `REG_RO(172, rx_error_counter_dn);
        `REG_RO(176, aurora_link_events_up);
        `REG_RO(180, aurora_link_events_dn);

        //tie in the virtual uart bus
        `REG_RO(204, {uart_rx_msg_tvalid, uart_tx_msg_tready, 14'b0, uart_rx_msg_tdata});
        `REG_STB(204, `stb_rd, uart_rx_msg_tready);
        `REG_WO(204, uart_tx_msg_tdata, '0);
        `REG_STB(204, `stb_wr, uart_tx_msg_tvalid);

        //bus test registers
        `REG_RW('hd0, icnt_test_stat , icnt_test_ctrl, 'h0); //208

        //lms7 spi
        `REG_RW(216, lms7_spi_rd_data, lms7_spi_wr_data, 'h0);
        `REG_STB(216, `stb_wr, lms7_spi_wr_valid);
        `REG_STB(216, `stb_rd, lms7_spi_rd_ready);
        `REG_RO(220, {lms7_spi_rd_valid, lms7_spi_wr_ready});
        `REG_WO(220, lms7_prog_spi_wr_addr, 16'b0);
        `REG_WO(224, lms7_prog_spi_wr_data, 32'b0);
        `REG_STB(224, `stb_wr, lms7_prog_spi_wr_valid);
        `REG_WO(228, lms7_spi_pattern, 24'b0);

        // AGC
        `REG_WO(232, agc_en, 32'b0);
        `REG_WO(236, agc_reset, 32'b0);
        `REG_WO(240, iq_thresh, 32'b0);
        `REG_WO(244, num_samps_sat, 32'b0);
        `REG_WO(248, max_num_samps_agc, 32'b0);
        `REG_WO(252, rssi_target, 32'b0);
        `REG_WO(256, wait_count_thresh, 32'b0);
        `REG_WO(260, small_jump, 8'b0);
        `REG_WO(264, big_jump, 8'b0);
        `REG_WO(268, test_gain_settings, 32'b0);
        `REG_WO(272, gain_lna_in, 32'b0);
        `REG_WO(276, gain_tia_in, 32'b0);
        `REG_WO(280, gain_pga_in, 32'b0);

        // RSSI
        //`REG_RO(284, meas_rssi);
        `REG_RO(284, agc_reg_tick_out_sys[31:0]);    // meas_rssi

        // Packet Detection
        `REG_WO(288, pkt_det_thresh, 32'b0);
        `REG_WO(292, pkt_det_num_samps, 16'b0);
        `REG_WO(296, pkt_det_en, 1'b0);
        `REG_WO(300, new_frame_flag, 1'b0);

        // CBRS Gains
        `REG_WO(304, gain_attn_in, 32'b0);
        `REG_WO(308, gain_lna1_in, 32'b0);
        `REG_WO(312, gain_lna2_in, 32'b0);

        // Initial gain setting
        `REG_WO(316, init_gain_in, 32'b0);

        // Number of coarse and fine tune stages
        `REG_WO(320, num_sat_stages_in, 32'b0);
        `REG_WO(324, num_fine_stages_in, 32'b0);
        //PRBS control registers
        `REG_WO(328, { trx_prbs_ctrl}, 16'h0);
        `REG_RO(332, trx_prbs_stat);
        `REG_RO(336, trx_prbs_e[31:0]);
        `REG_RO(340, trx_prbs_e[63:32]);
        //344 used by prbs
        `REG_RO('h344, icnt_test_em); //344 out of order
        //AGC regs
        `REG_RW(356, agc_status, agc_ctrl, 32'h0000);
        `REG_STB(356, `stb_wr, agc_ctrl_stb);
        `REG_RW(360, agc_table_read_data, agc_table_cmd_data, 32'h0000);
        `REG_STB(360, `stb_wr, agc_table_wr_stb);
    end

    xpm_cdc_single cdc_sys_trigger (
        .src_clk  (DATA_clk   ), .src_in   ( internal_trigger ),
        .dest_clk (SYS_clk    ), .dest_out ( SYS_trigger  ));

    ////////////////////////////////////////////////////////////////////
    // lms7 advanced spi control
    ////////////////////////////////////////////////////////////////////  
    reg SYS_tx_active_d;
    always @(posedge SYS_clk) SYS_tx_active_d <= SYS_tx_active;

    lms7_spi_master #(
        .PROG_SIZE(64), //number of entries (16 per trigger)
        .NUM_TRIGGERS(4)
    ) lms_spi_master
    (
        .clk(SYS_clk),
        .rst(SYS_rst || lms_dig_rst),

        //custom spi clock pattern for testing
        .wclk_pattern(lms7_spi_pattern[7:0]),
        .rclk_pattern(lms7_spi_pattern[15:8]),
        .miso_pattern(lms7_spi_pattern[23:16]),

        //event triggers
        .triggers_in({
            1'b0,
            !SYS_tx_active && SYS_tx_active_d,      //trig2 is tx disabled event
            SYS_tx_active && !SYS_tx_active_d,      //trig1 is tx enabled event
            SYS_trigger}),                          //trig0 is internal trigger

        //automatic gain controls
        /*
        .gain_override(gain_override),
        .gain_lna(gain_lna),
        .gain_tia(gain_tia),
        .gain_pga(gain_pga),
        .gain_pad(gain_pad),
        */
        .gain_override( agc_reg_tick_out_sys[86] ),
        .gain_lna( {agc_reg_tick_out_sys[85:82], agc_reg_tick_out_sys[85:82]} ),   // Currently we only care about chainA, just repeat...
        .gain_tia( {agc_reg_tick_out_sys[81:80], agc_reg_tick_out_sys[81:80]} ),
        .gain_pga( {agc_reg_tick_out_sys[79:68], agc_reg_tick_out_sys[79:68]} ),

        //program the spi state machine
        .prog_spi_wr_valid(lms7_prog_spi_wr_valid),
        .prog_spi_wr_addr(lms7_prog_spi_wr_addr),
        .prog_spi_wr_data(lms7_prog_spi_wr_data),

        //SPI master interface for lms
        .spi_mosi(lms_spi_sdio),
        .spi_miso(lms_spi_sdo),
        .spi_sclk(lms_spi_sclk),
        .spi_cs_n(lms_spi_sen),

        //transaction interface
        .user_spi_wr_valid(lms7_spi_wr_valid),
        .user_spi_wr_ready(lms7_spi_wr_ready),
        .user_spi_wr_data(lms7_spi_wr_data),

        //async readback for register reads
        .user_spi_rd_valid(lms7_spi_rd_valid),
        .user_spi_rd_ready(lms7_spi_rd_ready),
        .user_spi_rd_data(lms7_spi_rd_data)

        // debug - OBCH
        //.st_out_lms_spi(st_out_lms_spi)

    );

    wire tdd_trigger_out;
    xpm_cdc_single tx_gain_ctrl_tick (
        .src_clk  (DATA_clk   ), .src_in   ( tdd_trigger_out ),
        .dest_clk (SYS_clk    ), .dest_out (SYS_RFC_trigger  ));

    tx_gain_ctrl u_tx_gain_ctrl(
        .clk(SYS_clk),
        .rst(SYS_rst),
        .en(tx_gain_ctrl_en),
        .tick(SYS_RFC_trigger),
        .start_gain(tx_gain_start),
        .stop_gain(tx_gain_stop),
        .step_gain(tx_gain_step),
        .cur_gain(cur_gain_pad));

    ////////////////////////////////////////////////////////////////////
    // standard system pulses
    ////////////////////////////////////////////////////////////////////
    wire tick_1us, tick_100us, tick_10us, tick_1ms, tick_10ms, tick_100ms, tick_250ms, tick_1s;
    wire rapid_blink, breath_pulse;

    wire [2:0] breath_mode;

    std_pulses u_std_pulses(
        .clk         ( SYS_clk      ),
        .rst         ( SYS_rst | SYS_trigger_req_int ), //sync pulse also syncs pulses used for LEDs
        .tick_1us    ( tick_1us     ),
        .tick_10us   ( tick_10us    ),
        .tick_100us  ( tick_100us   ),
        .tick_1ms    ( tick_1ms     ),
        .tick_10ms   ( tick_10ms    ),
        .tick_100ms  ( tick_100ms   ),
        .tick_250ms  ( tick_250ms   ),
        .tick_1s     ( tick_1s      ),
        .hb_normal   ( /*open*/     ),
        .hb_error    ( /*open*/     ),
        .hb_alert    ( rapid_blink  ),
        .breath_mode ( breath_mode  ),
        .breath      ( breath_pulse )
    );

    ////////////////////////////////////////////////////////////////////
    // gpio register tie-ins
    ////////////////////////////////////////////////////////////////////
    assign gpio_in = {
        gpio_reg[9:7],
        clkbuff_irq,
        clkbuff_lock,
        gpio_reg[4:0]};

    //uart muxing for revc
    wire uart_tx, uart_rx;
    assign pmod_2  = rev_c? uart_tx : 1'bz; //tx serial output on revc, otherwise keep as input
    assign uart_rx = rev_c? pmod_3 : 1'b1; //pmod3 is rx on revc, otherwise zero for idle serial
    assign pmod_3  = rev_b? lms_hw_rstn : 1'bz; //lms rst on revb, uart rx on revc

    //support pgood_n to enable rev_c boards downstream of rev_b
    wire pgood_n_in    = 1'b0; //always good on iris
    wire pgood_n_out;
    assign pmod_7 = (rev_b) ? pgood_n_out : 1'bz;

    //LMS7002M Transceiver Control
    assign lms_hw_rstn     = gpio_reg[0];
    assign lms_dig_rst     = gpio_reg[1];
    assign lms_tdd_enb     = gpio_reg[9];
    assign lms_hw_rxen     = lms_tdd_enb?!SYS_tx_active:gpio_reg[2];
    assign lms_hw_txen     = lms_tdd_enb?SYS_tx_active:gpio_reg[3];

    //Clock Buffer Control
    assign clkbuff_rst_n   = !(gpio_reg[4] || SYS_rst); //desire this to be on by default

    //stay in boot state until gpios are written
    reg boot_st_enb = 1'b1;
    always @(posedge SYS_clk)
        if      (SYS_rst)    boot_st_enb <= 1'b1;
        else if (gpio_write) boot_st_enb <= 1'b0;

    wire bld_dirty = VERSION[31];
    wire bld_clean = !VERSION[31];
    assign breath_mode  = bld_dirty     ? 'd5 //dirty is coughing
                        : boot_st_enb   ? 'd2 //medium for boot
//                      : seq_dn_done   ? 'd1 //slow for shdn
//                      : st_main_wait  ? 'd3 //user holding button
//                      : spi_act       ? 'd3 //spi activity
                        : 'h1;                //fast at runtime


    // Save our Souls
    reg [29:0] sos = 30'b101010011011011001010100000000;
    always @(posedge SYS_clk) begin
        if (high_thermal_shutdown_event_kwd == 32'hea7) begin
            high_thermal_shutdown_event_accepted <= '1;
    end
        if (tick_100ms & high_thermal_shutdown_event_accepted) begin
            sos <= { sos[($bits(sos)-2):0], sos[$bits(sos)-1] };
    end

    end


    //LEDs
    wire activity = 1'b0;
    wire fpga_led_error  = high_thermal_shutdown_event_accepted ? sos[0] : ( blink_enable ? rapid_blink : gpio_reg[7] );
    wire fpga_led_good   =  ( boot_st_enb  ? breath_pulse
                              : blink_enable ? rapid_blink
                              : activity     ? breath_pulse
                              : gpio_reg[8]);

    reg sample_pgood;

    assign fpga_grn_led_n = ~fpga_led_good;
    assign fpga_red_led    = (rev_c & sample_pgood) ? 1'bz
                           :  rev_c                 ? fpga_led_error
                           : ~fpga_led_error;

    always@(posedge SYS_clk)
        if(SYS_rst)
            sample_pgood <= '0;
        else if (tick_250ms)
            sample_pgood <= '1;
        else if (tick_1ms & sample_pgood)
            sample_pgood <= '0;

    //fe_pgood_n comes in when the led is tristated
    always@(posedge SYS_clk)
        if(SYS_rst)
            fe_pgood <= '0;
        else if(tick_1ms & sample_pgood & rev_c)
            fe_pgood <= ~fpga_red_led;

    //monitor clock lock events
    reg clkbuff_lock_r;
    always @(posedge SYS_clk) clkbuff_lock_r <= clkbuff_lock;
    `EVENT_COUNTER(SYS_clk, SYS_rst, clock_events, clkbuff_lock_r != clkbuff_lock);

    ////////////////////////////////////////////////////////////////////
    // Iris RF core implementation
    ////////////////////////////////////////////////////////////////////
    `ifdef ENABLE_RRH
    iris_rfcore_rrh u_rfcore (
    `else
    iris_rfcore u_rfcore (
    `endif
        //clocks
        .SYS_clk            ( SYS_clk             ),
        .SYS_rst            ( SYS_rst             ),

        //triggers in
        .trigger_in     ( internal_trigger | RFC_trigger  ),
        .trigger_out    ( tdd_trigger_out ),

        //current time
        .SYS_time64(SYS_time64),

        //register interface
        .drpaddr_in (APB_M_paddr  [2]),
        .drpdi_in   (APB_M_pwdata [2]),
        .drpen_in   (APB_M_psel   [2] & APB_M_penable[2]),
        .drpwe_in   (APB_M_pwrite [2]),
        .drpdo_out  (APB_M_prdata [2]),
        .drprdy_out (APB_M_pready [2]),

        //data signals ( upper 16 I, lower 16 Q)
        .DATA_clk           ( DATA_clk           ),
        .DATA_rst           ( DATA_rst           ),
        .RX_data_a          ( RX_data_a          ),
        .RX_data_b          ( RX_data_b          ),
        .TX_active          ( TX_active          ),
        .TX_data_a          ( TX_data_a          ),
        .TX_data_b          ( TX_data_b          ),

        `ifdef ENABLE_RRH
        //beamformer
        .adder_clear             ( SYS_trigger_req_int ),
        .AXIS_RBF_RX_tdata       ( AXIS_RBF_RX_tdata ),
        .AXIS_RBF_RX_tlast       ( AXIS_RBF_RX_tlast ),
        .AXIS_RBF_RX_tready      ( AXIS_RBF_RX_tready),
        .AXIS_RBF_RX_tvalid      ( AXIS_RBF_RX_tvalid),
        .AXIS_RBF_TX_enabled     ( dnlink_active),
        .AXIS_RBF_TX_tdata       ( AXIS_RBF_TX_tdata ),
        .AXIS_RBF_TX_tlast       ( AXIS_RBF_TX_tlast ),
        .AXIS_RBF_TX_tready      ( AXIS_RBF_TX_tready),
        .AXIS_RBF_TX_tvalid      ( AXIS_RBF_TX_tvalid),
        .AXIS_RBF_TX_almost_full ( AXIS_RBF_TX_almost_full),
        .AXIS_TX_almost_full     ( AXIS_DSP_TX_almost_full),
        `endif

        //RF data fifos
        .AXIS_RX_tdata       ( AXIS_DSP_RX_tdata   ),
        .AXIS_RX_tkeep       ( AXIS_DSP_RX_tkeep   ),
        .AXIS_RX_tuser       ( AXIS_DSP_RX_tuser   ),
        .AXIS_RX_tlast       ( AXIS_DSP_RX_tlast   ),
        .AXIS_RX_tready      ( AXIS_DSP_RX_tready  ),
        .AXIS_RX_tvalid      ( AXIS_DSP_RX_tvalid  ),

        .AXIS_TX_tdata       ( AXIS_DSP_TX_tdata_buf),
        .AXIS_TX_tkeep       ( AXIS_DSP_TX_tkeep_buf),
        .AXIS_TX_tlast       ( AXIS_DSP_TX_tlast_buf),
        .AXIS_TX_tready      ( AXIS_DSP_TX_tready_buf),
        .AXIS_TX_tvalid      ( AXIS_DSP_TX_tvalid_buf));

    ////////////////////////////////////////////////////////////////////
    // Up and down links
    ////////////////////////////////////////////////////////////////////
    wire uart_rx_msg;

    iris030_up_dn_links u_up_dn_links (
        .local_hw_addr(local_hw_addr),
        .SYS_time64(SYS_time64),

        //clocks
        .SYS_clk            (SYS_clk),
        .SYS_rst            (SYS_rst),
        .SYS_stb            (tick_10us),
        .SYS_tak            (tick_1us),

        //uart
        .rxd((GT_UPL_squelch || GT_DNL_squelch)?1'b1:uart_tx), //prevent loops in testing
        .txd(uart_rx_msg),

        //virtual uart
        .uart_rx_msg_tdata(uart_rx_msg_tdata),
        .uart_rx_msg_tvalid(uart_rx_msg_tvalid),
        .uart_rx_msg_tready(uart_rx_msg_tready),
        .uart_tx_msg_tdata(uart_tx_msg_tdata),
        .uart_tx_msg_tvalid(uart_tx_msg_tvalid),
        .uart_tx_msg_tready(uart_tx_msg_tready),

        //interconnect mgts
        .mgtrefclk0_p(mgtrefclk0_p),
        .mgtrefclk0_n(mgtrefclk0_n),
        .mgttx_p(mgttx_p),
        .mgttx_n(mgttx_n),
        .mgtrx_p(mgtrx_p),
        .mgtrx_n(mgtrx_n),

        //test control interface
        .icnt_test_ctrl (icnt_test_ctrl),
        .icnt_test_stat (icnt_test_stat),
        .icnt_test_em   (icnt_test_em),

        //interconnect IOs
        .icnt_up_psnt_n_in  (icnt_up_psnt_n_in),
        .icnt_up_sclk       (icnt_up_sclk),
        .icnt_up_is_head_n  (icnt_up_is_head_n),
        .icnt_up_psnt_n_out (icnt_up_psnt_n_out),
        .icnt_up_sda        (icnt_up_sda),
        .icnt_up_pgood_n_in (pgood_n_in),
        .icnt_up_aux_clk_n  (icnt_up_aux_clk_n),
        .icnt_up_aux_clk_p  (icnt_up_aux_clk_p),
        .icnt_dn_psnt_n_in  (icnt_dn_psnt_n_in),
        .icnt_dn_sda        (icnt_dn_sda),
        .icnt_dn_pgood_n_out(pgood_n_out),
        .icnt_dn_psnt_n_out (icnt_dn_psnt_n_out),
        .icnt_dn_sclk       (icnt_dn_sclk),
        .icnt_dn_trigger    (icnt_dn_trigger),
        .icnt_dn_is_tail_n  (icnt_dn_is_tail_n),

        //i2c bus
        .I2C_scl_i          (I2C0_SCL_I),
        .I2C_scl_o          (I2C0_SCL_O),
        .I2C_scl_t          (I2C0_SCL_T),
        .I2C_sda_i          (I2C0_SDA_I),
        .I2C_sda_o          (I2C0_SDA_O),
        .I2C_sda_t          (I2C0_SDA_T),

        //triggers
        .SYS_incr_pulse_in(SYS_incr_pulse_in),
        .SYS_decr_pulse_in(SYS_decr_pulse_in),
        .SYS_sync_delays(SYS_sync_delays),
        .SYS_trigger_gen(SYS_trigger_gen),
        .SYS_send_ping_up(SYS_send_ping_up),
        .SYS_send_ping_dn(SYS_send_ping_dn),
        .SYS_trigger_status(SYS_msg_sync_status),
        .SYS_sync_done(SYS_sync_done),
        .SYS_trigger_req_int(SYS_trigger_req_int),
        .SYS_req_shutdown(SYS_req_shutdown),
        .SYS_req_restart(SYS_req_restart),
        .DATA_clk(DATA_clk),
        .DATA_rst(DATA_rst),
        .DATA_trigger_out(internal_trigger),

        //status/control/debugs
        .SYS_gt_ctrl_dn_stb(SYS_gt_ctrl_dn_stb),
        .SYS_gt_ctrl_up_stb(SYS_gt_ctrl_up_stb),
        .SYS_gt_ctrl_up(SYS_gt_ctrl_up),
        .SYS_gt_ctrl_dn(SYS_gt_ctrl_dn),
        .SYS_link_control(SYS_link_control),
        .SYS_link_status(SYS_link_status),
        .gtx_clk_counter(gtx_clk_counter),
        .rx_error_counter_up(rx_error_counter_up),
        .rx_error_counter_dn(rx_error_counter_dn),
        .upl_user_clk_cycles(upl_user_clk_cycles),
        .dnl_user_clk_cycles(dnl_user_clk_cycles),
        .UPL_link_active(uplink_active),
        .DNL_link_active(dnlink_active),

        //FIFO signals
        .AXIS_UPL_RX_tdata  ( AXIS_UPL_RX_tdata  ),
        .AXIS_UPL_RX_tlast  ( AXIS_UPL_RX_tlast  ),
        .AXIS_UPL_RX_tready ( AXIS_UPL_RX_tready ),
        .AXIS_UPL_RX_tvalid ( AXIS_UPL_RX_tvalid ),

        .AXIS_UPL_TX_tdata  ( AXIS_UPL_TX_tdata  ),
        .AXIS_UPL_TX_tlast  ( AXIS_UPL_TX_tlast  ),
        .AXIS_UPL_TX_tready ( AXIS_UPL_TX_tready ),
        .AXIS_UPL_TX_tvalid ( AXIS_UPL_TX_tvalid ),

        .AXIS_DNL_RX_tdata  ( AXIS_DNL_RX_tdata  ),
        .AXIS_DNL_RX_tlast  ( AXIS_DNL_RX_tlast  ),
        .AXIS_DNL_RX_tready ( AXIS_DNL_RX_tready ),
        .AXIS_DNL_RX_tvalid ( AXIS_DNL_RX_tvalid ),

        .AXIS_DNL_TX_tdata  ( AXIS_DNL_TX_tdata  ),
        .AXIS_DNL_TX_tlast  ( AXIS_DNL_TX_tlast  ),
        .AXIS_DNL_TX_tready ( AXIS_DNL_TX_tready ),
        .AXIS_DNL_TX_tvalid ( AXIS_DNL_TX_tvalid )
    );

    reg uplink_active_d, dnlink_active_d;
    always @(posedge SYS_clk) uplink_active_d <= uplink_active;
    always @(posedge SYS_clk) dnlink_active_d <= dnlink_active;
    `EVENT_COUNTER(SYS_clk, SYS_rst, aurora_link_events_up, uplink_active_d != uplink_active);
    `EVENT_COUNTER(SYS_clk, SYS_rst, aurora_link_events_dn, dnlink_active_d != dnlink_active);

    ////////////////////////////////////////////////////////////////////
    // axi stream router and ethernet
    ////////////////////////////////////////////////////////////////////
    wire dsp_pkt_drop;

    iris030_network u_network
    (
        //clocks
        .SYS_clk ( SYS_clk ),
        .SYS_rst ( SYS_rst ),

        //debug
        .pkt_drop(dsp_pkt_drop),

        //config bus
        .drpaddr_in (APB_M_paddr  [3]),
        .drpdi_in   (APB_M_pwdata [3]),
        .drpen_in   (APB_M_psel   [3] & APB_M_penable[3]),
        .drpwe_in   (APB_M_pwrite [3]),
        .drpdo_out  (APB_M_prdata [3]),
        .drprdy_out (APB_M_pready [3]),

        //eth config
        .local_hw_addr (local_hw_addr),

        //position in chain for beamformer destination
        .node_chain (SYS_msg_sync_status[31:28]),
        .node_index (SYS_msg_sync_status[27:24]),

        //flow control
        .GT_DNL_squelch(GT_DNL_squelch),
        .GT_UPL_squelch(GT_UPL_squelch),
        .NET_DMA_ready(NET_DMA_ready),
        .GEM_DMA_ready(GEM_DMA_ready),

        //up/dn stream fifos
        .AXIS_UPL_active    ( uplink_active ),
        .AXIS_UPL_TX_tdata  ( AXIS_UPL_TX_tdata  ),
        .AXIS_UPL_TX_tlast  ( AXIS_UPL_TX_tlast  ),
        .AXIS_UPL_TX_tvalid ( AXIS_UPL_TX_tvalid ),
        .AXIS_UPL_TX_tready ( AXIS_UPL_TX_tready ),
        .AXIS_UPL_RX_tdata  ( AXIS_UPL_RX_tdata  ),
        .AXIS_UPL_RX_tlast  ( AXIS_UPL_RX_tlast  ),
        .AXIS_UPL_RX_tvalid ( AXIS_UPL_RX_tvalid ),
        .AXIS_UPL_RX_tready ( AXIS_UPL_RX_tready ),

        .AXIS_DNL_active    ( dnlink_active ),
        .AXIS_DNL_TX_tdata  ( AXIS_DNL_TX_tdata  ),
        .AXIS_DNL_TX_tlast  ( AXIS_DNL_TX_tlast  ),
        .AXIS_DNL_TX_tvalid ( AXIS_DNL_TX_tvalid ),
        .AXIS_DNL_TX_tready ( AXIS_DNL_TX_tready ),
        .AXIS_DNL_RX_tdata  ( AXIS_DNL_RX_tdata  ),
        .AXIS_DNL_RX_tlast  ( AXIS_DNL_RX_tlast  ),
        .AXIS_DNL_RX_tvalid ( AXIS_DNL_RX_tvalid ),
        .AXIS_DNL_RX_tready ( AXIS_DNL_RX_tready ),

        //DSP streams
        .AXIS_DSP_RX_tdata       ( AXIS_DSP_RX_tdata ),
        .AXIS_DSP_RX_tkeep       ( AXIS_DSP_RX_tkeep ),
        .AXIS_DSP_RX_tuser       ( AXIS_DSP_RX_tuser ),
        .AXIS_DSP_RX_tlast       ( AXIS_DSP_RX_tlast ),
        .AXIS_DSP_RX_tready      ( AXIS_DSP_RX_tready),
        .AXIS_DSP_RX_tvalid      ( AXIS_DSP_RX_tvalid),

        .AXIS_DSP_TX_tdata       ( AXIS_DSP_TX_tdata ),
        .AXIS_DSP_TX_tkeep       ( AXIS_DSP_TX_tkeep ),
        .AXIS_DSP_TX_tlast       ( AXIS_DSP_TX_tlast ),
        .AXIS_DSP_TX_tready      ( AXIS_DSP_TX_tready),
        .AXIS_DSP_TX_tvalid      ( AXIS_DSP_TX_tvalid),
        .AXIS_DSP_TX_almost_full ( AXIS_DSP_TX_almost_full),

        //beamformer
        .AXIS_RBF_RX_tdata       ( AXIS_RBF_RX_tdata ),
        .AXIS_RBF_RX_tlast       ( AXIS_RBF_RX_tlast ),
        .AXIS_RBF_RX_tready      ( AXIS_RBF_RX_tready),
        .AXIS_RBF_RX_tvalid      ( AXIS_RBF_RX_tvalid),
        .AXIS_RBF_TX_tdata       ( AXIS_RBF_TX_tdata ),
        .AXIS_RBF_TX_tlast       ( AXIS_RBF_TX_tlast ),
        .AXIS_RBF_TX_tready      ( AXIS_RBF_TX_tready),
        .AXIS_RBF_TX_tvalid      ( AXIS_RBF_TX_tvalid),
        .AXIS_RBF_TX_almost_full ( AXIS_RBF_TX_almost_full),

        //AXI streams to and from the network stack DMA
        .AXIS_NET_S2MM_tdata  ( AXIS_NET_S2MM_tdata  ),
        .AXIS_NET_S2MM_tlast  ( AXIS_NET_S2MM_tlast  ),
        .AXIS_NET_S2MM_tready ( AXIS_NET_S2MM_tready ),
        .AXIS_NET_S2MM_tvalid ( AXIS_NET_S2MM_tvalid ),
        .AXIS_NET_MM2S_tdata  ( AXIS_NET_MM2S_tdata  ),
        .AXIS_NET_MM2S_tlast  ( AXIS_NET_MM2S_tlast  ),
        .AXIS_NET_MM2S_tready ( AXIS_NET_MM2S_tready ),
        .AXIS_NET_MM2S_tvalid ( AXIS_NET_MM2S_tvalid ),

        //AXI streams to and from the gigE mac DMA
        .AXIS_GEM_S2MM_tdata  ( AXIS_GEM_S2MM_tdata  ),
        .AXIS_GEM_S2MM_tlast  ( AXIS_GEM_S2MM_tlast  ),
        .AXIS_GEM_S2MM_tready ( AXIS_GEM_S2MM_tready ),
        .AXIS_GEM_S2MM_tvalid ( AXIS_GEM_S2MM_tvalid ),
        .AXIS_GEM_MM2S_tdata  ( AXIS_GEM_MM2S_tdata  ),
        .AXIS_GEM_MM2S_tlast  ( AXIS_GEM_MM2S_tlast  ),
        .AXIS_GEM_MM2S_tready ( AXIS_GEM_MM2S_tready ),
        .AXIS_GEM_MM2S_tvalid ( AXIS_GEM_MM2S_tvalid )
    );

    `EVENT_COUNTER(SYS_clk, SYS_rst, dsp_pkt_drops, dsp_pkt_drop);

    `ifdef ENABLE_RRH
    //bypass in RRH, large buffer is after beamformer
    //and we can handle packets coming in at line rate
    assign AXIS_DSP_TX_tdata_buf = AXIS_DSP_TX_tdata;
    assign AXIS_DSP_TX_tkeep_buf = AXIS_DSP_TX_tkeep;
    assign AXIS_DSP_TX_tlast_buf = AXIS_DSP_TX_tlast;
    assign AXIS_DSP_TX_tvalid_buf = AXIS_DSP_TX_tvalid;
    assign AXIS_DSP_TX_tready = AXIS_DSP_TX_tready_buf;
    `else
    //large fifo buffer to hold bursts of RF transmit data
    stream_fifo #(
        .DATA_WIDTH(128),
        .ENABLE_TKEEP(1),
        .FIFO_DEPTH(RF_TX_FIFO_DEPTH),
        .ALMOST_FULL_OFFSET(TX_BUFF_MTU_PKT))
    rf_tx_fifo
    (
        .clk ( SYS_clk ),
        .rst ( SYS_rst ),
        .in_tdata(AXIS_DSP_TX_tdata),
        .in_tkeep(AXIS_DSP_TX_tkeep),
        .in_tlast(AXIS_DSP_TX_tlast),
        .in_tvalid(AXIS_DSP_TX_tvalid),
        .in_tready(AXIS_DSP_TX_tready),
        .out_tdata(AXIS_DSP_TX_tdata_buf),
        .out_tkeep(AXIS_DSP_TX_tkeep_buf),
        .out_tlast(AXIS_DSP_TX_tlast_buf),
        .out_tvalid(AXIS_DSP_TX_tvalid_buf),
        .out_tready(AXIS_DSP_TX_tready_buf),
        .almost_empty(),
        .almost_full(AXIS_DSP_TX_almost_full)
    );
    `endif

    ////////////////////////////////////////////////////////////////////
    // LMS7 DIQ interface
    ////////////////////////////////////////////////////////////////////
    trxiq_top u_trxiq_top (
        .SYS_clk(SYS_clk),
        .SYS_rst(SYS_rst),

        .SYS_config_write(trxiq_config_write),
        .SYS_config_data(trxiq_config_data),

        .tick          (tick_1us      ),
        .SYS_prbs_ctrl (trx_prbs_ctrl ),
        .SYS_prbs_stat (trx_prbs_stat ),
        .SYS_prbs_e    (trx_prbs_e    ),

        .SYS_test_data_a_tx(trx_test_data_a_tx),
        .SYS_test_data_b_tx(trx_test_data_b_tx),
        .SYS_test_data_a_rx(trx_test_data_a_rx),
        .SYS_test_data_b_rx(trx_test_data_b_rx),
        .SYS_data_clk_counter(data_clk_counter),
        .DATA_clk(DATA_clk),
        .DATA_rst(DATA_rst),
        .RX_data_a(RX_data_a),
        .RX_data_b(RX_data_b),
        .TX_data_a(TX_data_a),
        .TX_data_b(TX_data_b),

        .LMS_DIQ2_IQSEL(lms_diq2_iqseldir),
        .LMS_DIQ2_MCLK(lms_diq2_mclk),
        .LMS_DIQ2_FCLK(lms_diq2_fclk),
        .LMS_DIQ2_TXNRX(lms_diq2_txnrx),
        .LMS_DIQ2_D(lms_diq2_d),

        .LMS_DIQ1_IQSEL(lms_diq1_iqseldir),
        .LMS_DIQ1_MCLK(lms_diq1_mclk),
        .LMS_DIQ1_FCLK(lms_diq1_fclk),
        .LMS_DIQ1_TXNRX(lms_diq1_txnrx),
        .LMS_DIQ1_D(lms_diq1_d)
    );

    ////////////////////////////////////////////////////////////////////
    // Automatic Gain Control (AGC) core implementation
    ////////////////////////////////////////////////////////////////////
 `ifndef ENABLE_RRH
    agc_core u_agc_core (
        .DATA_clk(DATA_clk),
        .DATA_rst(DATA_rst),
        .RX_data_a(RX_data_a),
        .RX_data_b(RX_data_b),

        // System clock
        .SYS_clk(SYS_clk),
        .SYS_rst(SYS_rst),

        .new_frame_flag( agc_reg_tick_data[3] || tdd_trigger_out ),

        //registers
        .cbrs_rev( agc_reg_tick_data[111:108] ),
        .agc_en( agc_reg_tick_data[0] ),
        .agc_reset(agc_reset),                      // translation inside agc core
        .iq_thresh( agc_reg_tick_data[35:4] ),
        .num_samps_sat( agc_reg_tick_data[43:36] ),
        .max_num_samps_agc( agc_reg_tick_data[51:44] ),
        .rssi_target( agc_reg_tick_data[83:52] ),
        .wait_count_thresh( wait_count_thresh ),    // sys clk domain

        .pkt_det_thresh( agc_reg_tick_data[205:174] ),
        .pkt_det_num_samps( agc_reg_tick_data[221:206] ),
        .pkt_det_en( agc_reg_tick_data[2] ),
        .init_gain_in( agc_reg_tick_data[173:166] ),

        .num_fine_stages_in( agc_reg_tick_data[227:225] ),
        .num_sat_stages_in( agc_reg_tick_data[224:222] ),

        //gain control
        .gain_override_out(rx_gain_ctrl_en),
        .gain_lna_out(gain_lna),
        .gain_tia_out(gain_tia),
        .gain_pga_out(gain_pga),
        .gain_lna1_out(gain_lna1),
        .gain_lna2_out(gain_lna2),
        .gain_attn_out(gain_attn),

        //rssi
        .meas_rssi_out(meas_rssi),

        //debug
        .small_jump( agc_reg_tick_data[99:92] ),
        .big_jump( agc_reg_tick_data[107:100] ),
        .test_gain_settings( agc_reg_tick_data[1] ),
        .gain_lna_in( agc_reg_tick_data[115:112] ),
        .gain_tia_in( agc_reg_tick_data[117:116] ),
        .gain_pga_in( agc_reg_tick_data[129:118] ),
        .gain_lna1_in( agc_reg_tick_data[133:130] ),
        .gain_lna2_in( agc_reg_tick_data[149:134] ),
        .gain_attn_in( agc_reg_tick_data[165:150] ),

        // Correlation trigger
        .corr_flag(RFC_trigger)
    );
`else
//temporary
    wire [ 3:0] agc_gain_lna_in  [0:1];
    wire [ 1:0] agc_gain_tia_in  [0:1];
    wire [11:0] agc_gain_pga_in  [0:1];
    wire        agc_gain_lna1_in [0:1];
    wire        agc_gain_lna2_in [0:1];
    wire [ 2:1] agc_gain_attn1_in [0:1];
    wire [ 2:1] agc_gain_attn2_in [0:1];
    assign  agc_gain_lna_in[0]  = '0;
    assign  agc_gain_tia_in[0]  = '0;
    assign  agc_gain_pga_in[0]  = '0;
    assign  agc_gain_lna1_in[0] = '0;
    assign  agc_gain_lna2_in[0] = '0;
    assign  agc_gain_attn1_in[0] = '0;
    assign  agc_gain_attn2_in[0] = '0;

    assign  agc_gain_lna_in[1]  = '0;
    assign  agc_gain_tia_in[1]  = '0;
    assign  agc_gain_pga_in[1]  = '0;
    assign  agc_gain_lna1_in[1] = '0;
    assign  agc_gain_lna2_in[1] = '0;
    assign  agc_gain_attn1_in[1] = '0;
    assign  agc_gain_attn2_in[1] = '0;

    assign agc_on = agc_status[31];
    assign agc_table_wr_en = agc_table_cmd_data[31];
    assign agc_table_addr  = agc_table_cmd_data[30:24];
    assign agc_table_data  = agc_table_cmd_data[23:0];
        // AGC core
        cpe_agc_core
        u_cpe_agc_core (
            // System clock
            .SYS_clk(SYS_clk),
            .SYS_rst(SYS_rst),
            .agc_trigger('0),
            .agc_table_wr_stb(agc_table_wr_stb),
            .agc_table_wr_en(agc_table_wr_en),
            .agc_table_addr(agc_table_addr),
            .agc_table_data(agc_table_data),
            .agc_table_read_data(agc_table_read_data),
            .agc_ctrl(agc_ctrl),
            .agc_ctrl_stb(agc_ctrl_stb),
            .agc_status(agc_status),
            .agc_en('0),
            .agc_reset(agc_reset),
            //gain control
            .gain_lna_out(gain_lna_out),
            .gain_tia_out(gain_tia_out),
            .gain_pga_out(gain_pga_out),
            .gain_lna1_out(gain_lna1_out),
            .gain_lna2_out(gain_lna2_out),
            .gain_attn1_out(gain_attn1_out),
            .gain_attn2_out(gain_attn2_out),

            // rssi
            .meas_rssi_a(meas_rssi_a),
            .meas_rssi_b(meas_rssi_b),

            .gain_lna_in(agc_gain_lna_in),
            .gain_tia_in(agc_gain_tia_in),
            .gain_pga_in(agc_gain_pga_in),
            .gain_lna1_in(agc_gain_lna1_in),
            .gain_lna2_in(agc_gain_lna2_in),
            .gain_attn1_in(agc_gain_attn1_in),
            .gain_attn2_in(agc_gain_attn2_in),

            .agc_debug(agc_debug)

        );
`endif
    ////////////////////////////////////////////////////////////////////
    // RF module control interface
    ////////////////////////////////////////////////////////////////////
    reg [16:1] rfmod_in;
    wire [16:1] rfmod_out, rfmod_oe;

    rfmod_iface u_rfmod
    (
        .clk(SYS_clk),
        .rst(SYS_rst),
        .sync(SYS_sync_done),
        .dut_pgood(1'b1), //always good on iris
        .tx_active({SYS_tx_active, SYS_tx_active}), //both channels controlled the same

        //RFMOD GPIO interface
        .rfmod_in(rfmod_in),
        .rfmod_out(rfmod_out),
        .rfmod_oe(rfmod_oe),

        //dont care detections
        .rfmod_id(),
        .id_valid(),

        // gain control
        /*
        .gain_override_out(rx_gain_ctrl_en),
        .agc_en(agc_en),
        .gain_lna1_out(gain_lna1),              // CBRS
        .gain_lna2_out(gain_lna2),              // CBRS
        .gain_attn_out(gain_attn),              // CBRS
        */

        .gain_override_out( agc_reg_tick_out_sys[86] ),
        .agc_en(agc_en),
        .gain_lna1_out( {agc_reg_tick_out_sys[67:64], agc_reg_tick_out_sys[67:64]} ),              // CBRS   Don't care about chainB at the moment
        .gain_lna2_out( {agc_reg_tick_out_sys[63:48], agc_reg_tick_out_sys[63:48]} ),              // CBRS
        .gain_attn_out( {agc_reg_tick_out_sys[47:32], agc_reg_tick_out_sys[47:32]} ),              // CBRS

        //JTAG programmer
        .dut_tdi(1'b0),
        .dut_tdo(),
        .dut_tck(1'b0),
        .dut_tms(1'b0),
        .dut_ten(1'b0),

        //register interface
        //4-byte addressing, even thought its 16 bit data
        .addr  (APB_M_paddr  [4][9:2]),
        .dati  (APB_M_pwdata [4][15:0]),
        .en    (APB_M_psel   [4] & APB_M_penable[4]),
        .wr    (APB_M_pwrite [4]),
        .dato  (APB_M_prdata [4][15:0]),
        .rdy   (APB_M_pready [4]),

        //external led controls
        //based on the state of the motherboard
        //On the test fe, these can be ignored.
        //On the iris, these are link indicators.
        .led_uplink_on(uplink_active),
        .led_dnlink_on(dnlink_active),
        .led_good(fpga_led_good),
        .led_error(fpga_led_error),

        .debug(),
        .cbrs_rev(cbrs_rev)
    );

    //connect RF mod GPIO bank to tri-states
    genvar ii;
    generate for (ii = 1; ii <= 16; ii = ii + 1) begin : gen_gpio
        assign rfmod_gpio[ii] = rfmod_oe[ii] ? rfmod_out[ii] : 1'bz;
        always @(posedge SYS_clk) rfmod_in[ii] <= rfmod_gpio[ii];
    end
    endgenerate

    xpm_cdc_single tx_active_in_sys (
        .src_clk  (DATA_clk   ), .src_in   (TX_active ),
        .dest_clk (SYS_clk    ), .dest_out (SYS_tx_active  ));

    ////////////////////////////////////////////////////////////////////
    // Pass in low speed interfaces - SPI, I2C, GPIO
    // Pass in source sync clock and data buses
    // Forward all DDR and FIXED IO
    ////////////////////////////////////////////////////////////////////
    `ifndef SIM
    design_1_wrapper u_zynq_ps (
        .DDR_addr          ( DDR_addr          ),
        .DDR_ba            ( DDR_ba            ),
        .DDR_cas_n         ( DDR_cas_n         ),
        .DDR_ck_n          ( DDR_ck_n          ),
        .DDR_ck_p          ( DDR_ck_p          ),
        .DDR_cke           ( DDR_cke           ),
        .DDR_cs_n          ( DDR_cs_n          ),
        .DDR_dm            ( DDR_dm            ),
        .DDR_dq            ( DDR_dq            ),
        .DDR_dqs_n         ( DDR_dqs_n         ),
        .DDR_dqs_p         ( DDR_dqs_p         ),
        .DDR_odt           ( DDR_odt           ),
        .DDR_ras_n         ( DDR_ras_n         ),
        .DDR_reset_n       ( DDR_reset_n       ),
        .DDR_we_n          ( DDR_we_n          ),
        .FIXED_IO_ddr_vrn  ( FIXED_IO_ddr_vrn  ),
        .FIXED_IO_ddr_vrp  ( FIXED_IO_ddr_vrp  ),
        .FIXED_IO_mio      ( FIXED_IO_mio      ),
        .FIXED_IO_ps_clk   ( FIXED_IO_ps_clk   ),
        .FIXED_IO_ps_porb  ( FIXED_IO_ps_porb  ),
        .FIXED_IO_ps_srstb ( FIXED_IO_ps_srstb ),

        //clocks
        .SYS_clk        ( SYS_clk           ),
        .SYS_rst        ( SYS_rst           ),

        .IRQ0           ( |(irqs0) ),
        .IRQ1           ( uart_rx_msg_tvalid ),

        .APB_M1_paddr   ( APB_M_paddr   [1] ),
        .APB_M1_penable ( APB_M_penable [1] ),
        .APB_M1_prdata  ( APB_M_prdata  [1] ),
        .APB_M1_pready  ( APB_M_pready  [1] ),
        .APB_M1_psel    ( APB_M_psel    [1] ),
        .APB_M1_pslverr ( APB_M_pslverr [1] ),
        .APB_M1_pwdata  ( APB_M_pwdata  [1] ),
        .APB_M1_pwrite  ( APB_M_pwrite  [1] ),

        .APB_M2_paddr   ( APB_M_paddr   [2] ),
        .APB_M2_penable ( APB_M_penable [2] ),
        .APB_M2_prdata  ( APB_M_prdata  [2] ),
        .APB_M2_pready  ( APB_M_pready  [2] ),
        .APB_M2_psel    ( APB_M_psel    [2] ),
        .APB_M2_pslverr ( APB_M_pslverr [2] ),
        .APB_M2_pwdata  ( APB_M_pwdata  [2] ),
        .APB_M2_pwrite  ( APB_M_pwrite  [2] ),

        .APB_M3_paddr   ( APB_M_paddr   [3] ),
        .APB_M3_penable ( APB_M_penable [3] ),
        .APB_M3_prdata  ( APB_M_prdata  [3] ),
        .APB_M3_pready  ( APB_M_pready  [3] ),
        .APB_M3_psel    ( APB_M_psel    [3] ),
        .APB_M3_pslverr ( APB_M_pslverr [3] ),
        .APB_M3_pwdata  ( APB_M_pwdata  [3] ),
        .APB_M3_pwrite  ( APB_M_pwrite  [3] ),

        .APB_M4_paddr   ( APB_M_paddr   [4] ),
        .APB_M4_penable ( APB_M_penable [4] ),
        .APB_M4_prdata  ( APB_M_prdata  [4] ),
        .APB_M4_pready  ( APB_M_pready  [4] ),
        .APB_M4_psel    ( APB_M_psel    [4] ),
        .APB_M4_pslverr ( APB_M_pslverr [4] ),
        .APB_M4_pwdata  ( APB_M_pwdata  [4] ),
        .APB_M4_pwrite  ( APB_M_pwrite  [4] ),

        //net dma
        .AXIS_NET_MM2S_tdata(AXIS_NET_MM2S_tdata),
        .AXIS_NET_MM2S_tkeep(/*open*/),
        .AXIS_NET_MM2S_tlast(AXIS_NET_MM2S_tlast),
        .AXIS_NET_MM2S_tvalid(AXIS_NET_MM2S_tvalid),
        .AXIS_NET_MM2S_tready(AXIS_NET_MM2S_tready),
        .AXIS_NET_S2MM_tdata(AXIS_NET_S2MM_tdata),
        .AXIS_NET_S2MM_tkeep('1),
        .AXIS_NET_S2MM_tlast(AXIS_NET_S2MM_tlast),
        .AXIS_NET_S2MM_tvalid(AXIS_NET_S2MM_tvalid),
        .AXIS_NET_S2MM_tready(AXIS_NET_S2MM_tready),

        //gem dma
        .AXIS_GEM_MM2S_tdata(AXIS_GEM_MM2S_tdata),
        .AXIS_GEM_MM2S_tkeep(/*open*/),
        .AXIS_GEM_MM2S_tlast(AXIS_GEM_MM2S_tlast),
        .AXIS_GEM_MM2S_tvalid(AXIS_GEM_MM2S_tvalid),
        .AXIS_GEM_MM2S_tready(AXIS_GEM_MM2S_tready),
        .AXIS_GEM_S2MM_tdata(AXIS_GEM_S2MM_tdata),
        .AXIS_GEM_S2MM_tkeep('1),
        .AXIS_GEM_S2MM_tlast(AXIS_GEM_S2MM_tlast),
        .AXIS_GEM_S2MM_tvalid(AXIS_GEM_S2MM_tvalid),
        .AXIS_GEM_S2MM_tready(AXIS_GEM_S2MM_tready),

        //i2c for sfp board
        .I2C0_SCL_I(I2C0_SCL_I),
        .I2C0_SCL_O(I2C0_SCL_O),
        .I2C0_SCL_T(I2C0_SCL_T),
        .I2C0_SDA_I(I2C0_SDA_I),
        .I2C0_SDA_O(I2C0_SDA_O),
        .I2C0_SDA_T(I2C0_SDA_T),
`ifdef CORR
        .coe_paddr   ( coe_paddr   ),
        .coe_penable ( coe_penable ),
        .coe_prdata  ( coe_prdata  ),
        .coe_pready  ( coe_pready  ),
        .coe_psel    ( coe_psel    ),
        .coe_pslverr ( coe_pslverr ),
        .coe_pwdata  ( coe_pwdata  ),
        .coe_pwrite  ( coe_pwrite  ),

//        // correlator debug,
//        .faros_corr_dbg_addr ( faros_corr_dbg_addr ),
//        .faros_corr_dbg_clk  ( faros_corr_dbg_clk  ),
//        .faros_corr_dbg_din  ( faros_corr_dbg_din  ),
//        .faros_corr_dbg_dout ( faros_corr_dbg_dout ),
//        .faros_corr_dbg_en   ( faros_corr_dbg_en   ),
//        .faros_corr_dbg_rst  ( faros_corr_dbg_rst  ),
//        .faros_corr_dbg_we   ( faros_corr_dbg_we   ),
//
//        // correlator debug,
//        .faros_corr_in_dbg_addr ( faros_corr_in_dbg_addr ),
//        .faros_corr_in_dbg_clk  ( faros_corr_in_dbg_clk  ),
//        .faros_corr_in_dbg_din  ( faros_corr_in_dbg_din  ),
//        .faros_corr_in_dbg_dout ( faros_corr_in_dbg_dout ),
//        .faros_corr_in_dbg_en   ( faros_corr_in_dbg_en   ),
//        .faros_corr_in_dbg_rst  ( faros_corr_in_dbg_rst  ),
//        .faros_corr_in_dbg_we   ( faros_corr_in_dbg_we   ),

`endif
        //uart for processor
        .UART1_rxd    ( uart_rx & uart_rx_msg ), //combine physical uart with msg sync one
        .UART1_txd    ( uart_tx      )

        /*  // LMS7 SPI MASTER - OBCH
        .gain_override(gain_override),
        .prog_spi_wr_valid(lms7_prog_spi_wr_valid),
        .user_spi_wr_valid(lms7_spi_wr_valid),
        .user_spi_wr_ready(lms7_spi_wr_ready),
        .user_spi_rd_valid(lms7_spi_rd_valid),
        .user_spi_rd_ready(lms7_spi_rd_ready),
        .st_out_lms_spi(st_out_lms_spi)   */
    );
    `endif

`ifdef CORR
    // MMCM configured for 5MHz here.
    // but we need a variable 8x clk
    MMCME2_BASE # (
        .CLKFBOUT_MULT_F  ( 24      ),
        .CLKOUT0_DIVIDE_F ( 3       ),
        .DIVCLK_DIVIDE    ( 1       ),
        .CLKIN1_PERIOD    ( 12.5000 ) //ns
    ) u_mmcm_sysgen (
        .CLKFBOUT ( FAROS_fbclk       ),
        .CLKOUT0  ( SYSGEN_clk        ),
        .CLKFBIN  ( FAROS_fbclk       ),
        .CLKIN1   ( DATA_clk          ),
        .PWRDWN   ( 0                 ),
        .RST      ( DATA_rst          ),
        .LOCKED   ( SYSGEN_dcm_locked )
    );

    xpm_cdc_sync_rst s_FAROS_rst_gen (
        .dest_clk ( DATA_clk),
        .src_rst  ( RFC_rst  ),
        .dest_rst ( FAROS_rst)
    );

    xpm_cdc_sync_rst s_SYSGEN_rst_gen (
        .dest_clk ( SYSGEN_clk),
        .src_rst  ( RFC_rst  ),
        .dest_rst ( SYSGEN_rst)
    );

    event_counter_cdc #(.WIDTH(32))
    SYSGEN_cycle_counter(
        .clk(SYSGEN_clk),
        .rst(SYSGEN_rst),
        .dest_clk(SYS_clk),
        .count(sysgen_clk_counter));

    event_counter_cdc #(.WIDTH(32))
    RFC_trigger_counter(
        .clk(DATA_clk),
        .rst(DATA_rst),
        .evt(RFC_trigger),
        .dest_clk(SYS_clk),
        .count(sync_counter));

    wire [31:0] FAROS_corr_en1x;
    reg corr_test_en;
    xpm_cdc_array_single # (4,0,0,1,0,32) s_corr_ctrl(SYS_clk, faros_corr_en, DATA_clk, FAROS_corr_en1x);
    wire [23:0] corr_data = FAROS_corr_en1x[5] ? (FAROS_corr_en1x[6] ? {corr_test_data[31:20], corr_test_data[15:4]} : {RX_data_b[31:20], RX_data_b[15:4]}) :{RX_data_a[31:20], RX_data_a[15:4]};
    wire [23:0] sig_corr_in = FAROS_corr_en1x[4] ? corr_data: 23'b0;
    wire sig_corr_v = FAROS_corr_en1x[6] ? corr_test_en : FAROS_corr_en1x[0];
    always @(posedge DATA_clk)
        if (FAROS_rst || FAROS_corr_en1x[6] == 1'b0) begin
            corr_test_addr <= '0;
            corr_test_en <= 1'b0;
        end else if (corr_test_addr < 1023) begin
            corr_test_addr <= corr_test_addr + 1'b1;
            corr_test_en <= FAROS_corr_en1x[4];
        end else
            corr_test_en <= 1'b0;

//    BRAM_TDP_MACRO #(
//        .BRAM_SIZE           ( "36Kb"        ) , // Target BRAM: "18Kb" or "36Kb"
//        .DEVICE              ( "7SERIES"     ) , // Target device: "7SERIES"
//        .DOA_REG             ( 0             ) , // Optional port A output register (0 or 1)
//        .DOB_REG             ( 0             ) , // Optional port B output register (0 or 1)
//        .INIT_FILE           ( "NONE"        ) ,
//        .READ_WIDTH_A        ( 32            ) , // Valid values are 1-36 (19-36 only valid when BRAM_SIZE="36Kb" )
//        .READ_WIDTH_B        ( 32            ) , // Valid values are 1-36 (19-36 only valid when BRAM_SIZE="36Kb" )
//        .SIM_COLLISION_CHECK ( "ALL"         ) , // Collision check enable "ALL", "WARNING_ONLY", "GENERATE_X_ONLY" or "NONE"
//        .WRITE_MODE_A        ( "WRITE_FIRST" ) , // "WRITE_FIRST", "READ_FIRST", or "NO_CHANGE"
//        .WRITE_MODE_B        ( "WRITE_FIRST" ) , // "WRITE_FIRST", "READ_FIRST", or "NO_CHANGE"
//        .WRITE_WIDTH_A       ( 32            ) , // Valid values are 1-36 (19-36 only valid when BRAM_SIZE="36Kb" )
//        .WRITE_WIDTH_B       ( 32            )   // Valid values are 1-36 (19-36 only valid when BRAM_SIZE="36Kb" )
//    ) u_dbg_ram (
//        //A-side to CPU for debug
//        .ADDRA  ( faros_corr_in_dbg_addr [11:2]) , // Input port-A address, width defined by Port A depth
//        .CLKA   ( faros_corr_in_dbg_clk    ) , // 1-bit input port-A clock
//        .DIA    ( faros_corr_in_dbg_din    ) , // Input port-A data, width defined by WRITE_WIDTH_A parameter
//        .DOA    ( faros_corr_in_dbg_dout   ) , // Output port-A data, width defined by READ_WIDTH_A parameter
//        .ENA    ( faros_corr_in_dbg_en     ) , // 1-bit input port-A enable
//        .REGCEA ( 1'b1       ) , // 1-bit input port-A output register enable
//        .RSTA   ( faros_corr_in_dbg_rst    ) , // 1-bit input port-A reset
//        .WEA    ( faros_corr_in_dbg_we     ) , // Input port-A write enable, width defined by Port A depth
//
//        //B-side debug write
//        .ADDRB  ( corr_test_addr ) , // Input port-B address, width defined by Port B depth
//        .CLKB   ( DATA_clk      ) , // 1-bit input port-B clock
//        .DIB    ( 1'b0           ) , // Input port-B data, width defined by WRITE_WIDTH_B parameter
//        .DOB    ( corr_test_data ) , // Output port-B data, width defined by READ_WIDTH_B parameter
//        .ENB    ( corr_test_en   ) , // 1-bit input port-B enable
//        .REGCEB ( 1'b1           ) , // 1-bit input port-B output register enable
//        .RSTB   ( FAROS_rst      ) , // 1-bit input port-B reset
//        .WEB    ( {4{1'b0}}      )   // Input port-B write enable, width defined by Port B depth
//    );

//    wire [36:0] corr_peak_abs;
//    wire [36:0] corr_thresh_scaled;
//    wire [17:0] sig_accum;
//    wire [23:0] sig_abs_sq;
//    wire [23:0] sig_abs_sq_dly;
//    wire [27:0] sig_accum_sq;
//    wire [27:0] sig_sq_accum;
//    wire [27:0] sig_stat;
//    wire [17:0] corr_accum;
//    wire [27:0] corr_accum_sq;
    
    wire [11:0] sig_delay_i;
    wire [11:0] sig_delay_q;
    
    xpm_cdc_array_single # (4,0,0,1,0,32) s_thresh_ctrl(SYS_clk, threshold_scale, DATA_clk, threshold_scale_1x);

    faros_corr  #(
        .CORR_SZ      (128),
        .INPUT_SZ     (12),
        .CORR_RES_SZ  (32),
        .RES_TRUNC_SZ (18),
        .PEAK_RES_SZ  (37),
        .COEFF_RD_BACK("FALSE"),
        .CONJ_COEFF   ("FALSE")
     ) u_faros_corr (
         .sysclk           ( SYS_clk ),
         .sysrst           ( SYS_rst ),
         .enable           ( faros_corr_en[0] ),

         .coe_penable      ( coe_penable ),
         .coe_psel         ( coe_psel ),
         .coe_paddr        ( coe_paddr ),
         .coe_pwdata       ( coe_pwdata ),
         .coe_prdata       ( coe_prdata ),
         .coe_pready       ( coe_pready ),
         .coe_pslverr      ( coe_pslverr ),
         .coe_pwrite       ( coe_pwrite ),

//         //debug ram interface (sysclk)
//         .dbg_addr ( {22'b0, faros_corr_dbg_addr[11:2]}),
//         .dbg_din  ( faros_corr_dbg_din ),
//         .dbg_dout ( faros_corr_dbg_dout ),
//         .dbg_en   ( faros_corr_dbg_en ),
//         .dbg_we   ( faros_corr_dbg_we ),
//         .dbg_sel  ( faros_corr_en[15:12] ),

         //Sample I/Q Interface (1x sample clock)
         .clk_1x           ( DATA_clk ),
         .rst_1x_i         ( FAROS_rst ),

         .sig_v            ( sig_corr_v ),
         .sig_i            ( sig_corr_in[23:12] ),
         .sig_q            ( sig_corr_in[11:0]  ),
         
         .thresh_scale     ( threshold_scale_1x[11:0]),
         
//         .sig_delay_i      ( sig_delay_i ),
//         .sig_delay_q      ( sig_delay_q ),
//
//         //corr_peak_v      out std_logic,
//         .corr_delay_i     ( faros_corr_0_delayed_i ),
//         .corr_delay_q     ( faros_corr_0_delayed_q ),
//
//          //corr_v           out std_logic,
//         .corr_i           ( faros_corr_0_i ),
//         .corr_q           ( faros_corr_0_q ),

//         .sig_accum         ( sig_accum         ),
//         .sig_abs_sq        ( sig_abs_sq        ),
//         .sig_abs_dly_sq    ( sig_abs_sq_dly    ),
//         .sig_accum_sq_out  ( sig_accum_sq      ),
//         .sig_sq_accum      ( sig_sq_accum      ),
//         .sig_stat          ( sig_stat          ),
//         .corr_accum_trunc  ( corr_accum        ),
//         .corr_accum_sq_out ( corr_accum_sq     ),
//         .corr_thresh_scaled( corr_thresh_scaled),
//         .corr_peak_abs     ( corr_peak_abs     ),
         .sync              ( RFC_trigger       ),

         //Correlator Internals (8x sample clock)
         .clk_8x           ( SYSGEN_clk ),
         .rst_8x_i         ( SYSGEN_rst )
         //rst_8x_o out std_logic,
     );

/*     
ila_0 (
    .clk(SYSGEN_clk),
    .probe0({sig_corr_in[23:12]}), 
    .probe1({sig_corr_in[11:0] }),
    .probe2(sig_delay_i),
    .probe3(sig_delay_q),
    .probe4(sig_abs_sq),
    .probe5(sig_abs_sq_dly),
    .probe6({sig_accum}),
    .probe7({sig_accum_sq}),
    .probe8({sig_sq_accum}),
    .probe9({faros_corr_0_i}), 
    .probe10({faros_corr_0_q}),
    .probe11(faros_corr_0_delayed_i),
    .probe12(faros_corr_0_delayed_q),
    .probe13({corr_accum}),
    .probe14({corr_accum_sq}),
    .probe15({corr_thresh_scaled}), 
    .probe16({corr_peak_abs}),
    .probe17(corr_test_en),
    .probe18(RFC_trigger),
    .probe19(DATA_clk)
);  
*/
`endif
endmodule

`default_nettype wire
