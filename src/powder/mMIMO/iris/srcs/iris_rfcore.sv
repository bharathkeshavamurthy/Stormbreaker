//----------------------------------------------------------------------------------
//-- Iris RF Core
//
//-- Controls streaming between raw IQ interfaces and framed fifo signals.
//
//-- The Rx framer accepts control requests to frame RX DIQ samples into packets
//-- and can operate based on trigger IOs and timestamp events.
//
//-- The Tx deframer accepts framed stream packets and transmits them over DIQ
//-- based on trigger IOs and timestamp events specified by the packet headers.
//----------------------------------------------------------------------------------

`default_nettype none

module iris_rfcore #(
    parameter REPLAY_RAM_SIZE = 4*1024,
    parameter BEACON_RAM_SIZE = 512)
(
    //clocks
    input wire SYS_clk,
    input wire SYS_rst,

    //triggers in
    input wire trigger_in,
    output wire trigger_out,

    //most recent time in sys clock domain
    output wire [63:0] SYS_time64,

    //APB register interface
//  input wire         drpclk_in,
    input wire [31:0]  drpaddr_in,
    input wire [31:0]  drpdi_in,
    input wire         drpen_in,
    input wire         drpwe_in,
    output wire [31:0] drpdo_out,
    output wire        drprdy_out,

    //data signals (upper 16 I, lower 16 Q)
    input wire DATA_clk,
    input wire DATA_rst,
    input wire [31:0] RX_data_a,
    input wire [31:0] RX_data_b,
    output reg TX_active,
    output wire [31:0] TX_data_a,
    output wire [31:0] TX_data_b,

    //DMA AXIS fifos
    output wire [127:0] AXIS_RX_tdata,
    output wire [15:0] AXIS_RX_tkeep,
    output wire [0:0] AXIS_RX_tuser,
    output wire AXIS_RX_tlast,
    input wire AXIS_RX_tready,
    output wire AXIS_RX_tvalid,

    input wire [127:0] AXIS_TX_tdata,
    input wire [15:0] AXIS_TX_tkeep,
    input wire AXIS_TX_tlast,
    output wire AXIS_TX_tready,
    input wire AXIS_TX_tvalid
);

    //system time tracking
    reg SYS_time_event;
    reg SYS_time_dummy;
    reg SYS_time_write;
    reg SYS_time_asap;
    reg SYS_time_read;
    reg [63:0] SYS_time_next;
    wire [63:0] SYS_time_last;
    wire [63:0] time_now;

    //status bus back through rx path
    wire [127:0] AXIS_STAT_tdata;
    wire AXIS_STAT_tlast;
    wire AXIS_STAT_tready;
    wire AXIS_STAT_tvalid;

    //rx control
    reg in_control_valid;
    wire in_control_ready;
    reg in_control_wait_trigger;
    reg in_control_has_time;
    reg [63:0] in_control_time;
    reg in_control_is_burst;
    reg [15:0] in_control_num_samps;
    reg [15:0] rx_frame_size;

    //control
    reg tx_clear, rx_clear;

    //tx status
    wire out_status_valid;
    wire out_status_ready;
    wire out_status_end_burst;
    wire out_status_underflow;
    wire out_status_time_late;
    wire out_status_has_time;
    wire [63:0] out_status_time;

    //tx sequence for flow control
    wire [15:0] sequence_tdata;
    wire sequence_terror;
    wire sequence_tvalid;
    wire sequence_tready;

    //data format
    reg [7:0] rx_packer_ctrl;
    reg [7:0] tx_unpacker_ctrl;

    //more tx path signals
    wire DAC_tx_active;
    reg SYS_tx_enb_override;
    reg [15:0] SYS_tx_enable_delay;
    reg [15:0] SYS_tx_disable_delay;

    //rx snooper
    wire [31:0] rx_snooper_data;
    wire rx_snooper_valid;
    reg rx_snooper_rd, rx_snooper_clr, rx_snooper_ch;

    //tx ram
    reg tx_replay_enable;
    reg [15:0] tx_replay_start, tx_replay_stop;
    reg tx_ram_write_stb [0:1];
    reg [15:0] tx_ram_write_addr;
    reg [31:0] tx_ram_write_data [0:1];

    // beacon ram
    reg [15:0] SYS_beacon_symbol;
    reg [15:0] SYS_beacon_stop;
    reg [15:0] SYS_beacon_start;
    reg [8:0] SYS_beacon_addr;
    reg [31:0] SYS_beacon_din;
    reg SYS_beacon_stb;

    //tdd frame schedule controls
    wire [63:0] SYS_time_last_tdd;
    wire [63:0] SYS_time_last_fdd;
    reg [31:0] SYS_tdd_conf;
    reg        SYS_tdd_stb;
    reg [15:0] SYS_tdd_samples_per_symbol;
    reg [15:0] SYS_tdd_symbols_per_frame;
    reg [31:0] SYS_tdd_frame_count_max;

    reg [11:0] SYS_sched_addr;
    reg [3:0] SYS_sched_din;
    reg SYS_sched_stb;
    assign SYS_time_last = SYS_tdd_conf[31] == 1'b1 ? SYS_time_last_tdd : SYS_time_last_fdd;

    reg [6:0] SYS_weight_max_addr;
    reg       SYS_weighted_beacon;
    reg [6:0] SYS_weight_addr;
    reg [31:0] SYS_weight_din [0:1];
    reg SYS_weight_stb [0:1];

    wire DATA_tdd_mode;
    wire rx_tdd_eob;
    wire rx_tdd_active, txram_tdd_active;
    wire DAC_tx_from_ram;

    //--------------------------------------------------------------------
    //-- register interface for controls
    //--------------------------------------------------------------------
    `include "system_regs.vh"

    `REG_INIT(SYS_rst, 32'hff, drpaddr_in, drpen_in, drpwe_in, drpdi_in, drpdo_out, drprdy_out);
    always @(posedge SYS_clk) begin
        `REG_READY_DEFAULT;

        //time master control
        `REG_RW(16, SYS_time_last[31:0], SYS_time_next[31:0], 0);
        `REG_RW(20, SYS_time_last[63:32], SYS_time_next[63:32], 0);
        `REG_WO(24, {SYS_time_read, SYS_time_asap, SYS_time_write, SYS_time_dummy}, 4'b0);
        `REG_STB(24, `stb_wr, SYS_time_event);

        //controls
        `REG_WO(28, tx_clear, '0);
        `REG_WO(32, rx_clear, '0);

        //rx framer control bus
        `REG_WO(48, in_control_time[31:0], '0);
        `REG_WO(52, in_control_time[63:32], '0);
        `REG_RW(56, in_control_ready, {in_control_wait_trigger, in_control_has_time, in_control_is_burst, in_control_num_samps}, '0);
        `REG_STB(56, `stb_wr, in_control_valid);

        //data packing configuration
        `REG_WO(68, rx_packer_ctrl, 0);
        `REG_WO(72, tx_unpacker_ctrl, 0);
        `REG_WO(76, rx_frame_size, '0);

        //transmit switch controls
        `REG_WO(80, SYS_tx_enable_delay, 16'b0);
        `REG_WO(84, SYS_tx_disable_delay, 16'b0);
        `REG_RW(88, SYS_tx_enb_override, SYS_tx_enb_override, 1'b0);

        //rx snooper ties ins
        `REG_RO(92, rx_snooper_data);
        `REG_STB(92, `stb_rd, rx_snooper_rd);
        `REG_RW(96, rx_snooper_valid, {rx_snooper_ch, rx_snooper_clr}, 2'b0); //st/ctrl

        //tx ram controls
        `REG_WO(100, tx_replay_enable, '0);
        `REG_WO(104, {tx_replay_start, tx_replay_stop}, '0);
        `REG_WO(108, tx_ram_write_addr, '0);
        `REG_WO(112, tx_ram_write_data[0], '0);
        `REG_WO(116, tx_ram_write_data[1], '0);
        `REG_STB(112, `stb_wr, tx_ram_write_stb[0]);
        `REG_STB(116, `stb_wr, tx_ram_write_stb[1]);

        // tdd mode
        // conf bitmap
        // bit 31: tdd=1 or fdd=0
        // bit 30: trig_out_en
        // bit 29: wait_trigger
        // bit 28: two pilots perframe
        // bit 27:24 max_frame
        `REG_WO(120, SYS_tdd_conf[31:0], '0);
        `REG_STB(120, `stb_wr, SYS_tdd_stb);
        `REG_WO(124, SYS_tdd_samples_per_symbol, '0);
        `REG_WO(128, SYS_tdd_symbols_per_frame, '0);
        `REG_WO(132, SYS_tdd_frame_count_max, '0);
        `REG_WO(136, SYS_sched_addr, 12'b0);
        `REG_WO(140, SYS_sched_din, 4'b0);
        `REG_STB(140, `stb_wr, SYS_sched_stb);
        `REG_WO(144, SYS_weight_addr, 7'b0);
        `REG_WO(148, SYS_weight_din[0], 32'b0);
        `REG_WO(152, SYS_weight_din[1], 32'b0);
        `REG_STB(148, `stb_wr, SYS_weight_stb[0]);
        `REG_STB(152, `stb_wr, SYS_weight_stb[1]);
        `REG_WO(156, {SYS_weighted_beacon, SYS_weight_max_addr}, 8'b0);
        `REG_WO(160, SYS_beacon_symbol, 16'b0);
        `REG_WO(164, SYS_beacon_addr, 9'b0);
        `REG_WO(168, SYS_beacon_din, 32'b0);
        `REG_STB(168, `stb_wr, SYS_beacon_stb);
        `REG_WO(172, {SYS_beacon_stop, SYS_beacon_start}, 32'b0);
    end

    //--------------------------------------------------------------------
    //-- TX active switch control and programmable delays
    //--------------------------------------------------------------------
    wire [15:0] DAC_enable_delay;
    wire [15:0] DAC_disable_delay;
    wire DAC_tx_enb_override;
    xpm_cdc_array_single #(.WIDTH(33)) tx_delay_xfer(
        .src_clk(SYS_clk), .src_in({SYS_tx_enb_override, SYS_tx_enable_delay, SYS_tx_disable_delay}),
        .dest_clk(DATA_clk), .dest_out({DAC_tx_enb_override, DAC_enable_delay, DAC_disable_delay}));

    reg [15:0] DAC_delay_count;
    wire TX_active_int = DAC_tx_active | DAC_tx_from_ram | DAC_tx_enb_override;
    always @(posedge DATA_clk)
        if (DATA_rst) DAC_delay_count <= '0;
        else if (TX_active_int == TX_active) DAC_delay_count <= '0;
        else DAC_delay_count <= DAC_delay_count + 1'b1;

    wire DAC_ramp_timeout = (!TX_active && DAC_delay_count == DAC_enable_delay) ||
                            ( TX_active && DAC_delay_count == DAC_disable_delay);

    always @(posedge DATA_clk)
        if (DATA_rst)              TX_active <= '0;
        else if (DAC_ramp_timeout) TX_active <= TX_active_int;

    //--------------------------------------------------------------------
    //-- rx stream snooper
    //--------------------------------------------------------------------
    wire DATA_rx_snooper_clr, DATA_rx_snooper_ch;
    xpm_cdc_array_single #(.WIDTH(2)) u_rx_snooper_cdc (
        .src_clk(SYS_clk), .src_in({rx_snooper_ch, rx_snooper_clr}),
        .dest_clk(DATA_clk), .dest_out({DATA_rx_snooper_ch, DATA_rx_snooper_clr}));

    cdc_stream_fifo #(.DATA_WIDTH(32), .FIFO_DEPTH(1024))
    RX_stream_snooper(
        .in_clk(DATA_clk),
        .in_rst(DATA_rx_snooper_clr),
        .in_tdata(DATA_rx_snooper_ch?RX_data_b:RX_data_a),
        .in_tlast(1'b0),
        .in_tvalid(1'b1),
        .in_tready(),
        .in_tkeep('1),
        .out_clk(SYS_clk),
        .out_rst(rx_snooper_clr),
        .out_tdata(rx_snooper_data),
        .out_tlast(),
        .out_tvalid(rx_snooper_valid),
        .out_tready(rx_snooper_rd),
        .out_tkeep(), .almost_empty(), .almost_full());

    //--------------------------------------------------------------------
    //-- cross domains between fclk and rx
    //--------------------------------------------------------------------
    wire [63:0] AXIS_RX_tdata_i0;
    wire AXIS_RX_tlast_i0;
    wire AXIS_RX_tvalid_i0;
    wire AXIS_RX_tready_i0;
    wire [63:0] AXIS_RX_tdata_i1;
    wire [7:0] AXIS_RX_tkeep_i1;
    wire AXIS_RX_tlast_i1;
    wire AXIS_RX_tvalid_i1;
    wire AXIS_RX_tready_i1;
    wire [127:0] AXIS_RX_tdata_i2;
    wire [15:0] AXIS_RX_tkeep_i2;
    wire AXIS_RX_tlast_i2;
    wire AXIS_RX_tvalid_i2;
    wire AXIS_RX_tready_i2;

    wire [63:0] out_sample_data;
    wire out_sample_valid;
    wire out_sample_ready;
    wire out_sample_end_burst;
    wire out_sample_error;
    wire out_sample_overflow;
    wire out_sample_time_late;
    wire out_sample_time_event;
    wire out_sample_trig_event;
    wire out_sample_has_time;
    wire [63:0] out_sample_time;
    wire out_sample_in_burst;
    wire [15:0] out_sample_countdown;

    twbw_rx_bus #(.DATA_WIDTH(64), .TIME_WIDTH(64)) u_rx_bus (
        //system clock domain
        .SYS_clk(SYS_clk),
        .SYS_rst(SYS_rst || rx_clear),

        //rx sample interface - sys clock domain
        .out_sample_data(out_sample_data),
        .out_sample_valid(out_sample_valid),
        .out_sample_ready(out_sample_ready),
        .out_sample_end_burst(out_sample_end_burst),
        .out_sample_error(out_sample_error),
        .out_sample_overflow(out_sample_overflow),
        .out_sample_time_late(out_sample_time_late),
        .out_sample_time_event(out_sample_time_event),
        .out_sample_trig_event(out_sample_trig_event),
        .out_sample_has_time(out_sample_has_time),
        .out_sample_time(out_sample_time),
        .out_sample_in_burst(out_sample_in_burst),
        .out_sample_countdown(out_sample_countdown),

        //control interface - sys clock domain
        .in_control_valid(in_control_valid),
        .in_control_ready(in_control_ready),
        .in_control_wait_trigger(in_control_wait_trigger),
        .in_control_has_time(in_control_has_time),
        .in_control_time(in_control_time),
        .in_control_is_burst(in_control_is_burst),
        .in_control_num_samps(in_control_num_samps),

        //rx clock domain interface (rx sample on each strobe cycle)
        .RX_clk(DATA_clk),
        .RX_rst(DATA_rst),
        .RX_stb(DATA_tdd_mode?rx_tdd_active:1'b1),
        .RX_eob(DATA_tdd_mode?rx_tdd_eob:1'b0),
        .RX_active(/*open*/),
        .RX_data({RX_data_b, RX_data_a}),

        //input events in rx clock domain
        .in_trigger(trigger_in),
        .in_time(time_now));

    twbw_rx_framer64 u_framer (
        .clk(SYS_clk),
        .rst(SYS_rst),
        .frame_size(rx_frame_size),
        .out_tdata(AXIS_RX_tdata_i0),
        .out_tlast(AXIS_RX_tlast_i0),
        .out_tvalid(AXIS_RX_tvalid_i0),
        .out_tready(AXIS_RX_tready_i0),
        .in_sample_data(out_sample_data),
        .in_sample_valid(out_sample_valid),
        .in_sample_ready(out_sample_ready),
        .in_sample_end_burst(out_sample_end_burst),
        .in_sample_error(out_sample_error),
        .in_sample_overflow(out_sample_overflow),
        .in_sample_time_late(out_sample_time_late),
        .in_sample_time_event(out_sample_time_event),
        .in_sample_trig_event(out_sample_trig_event),
        .in_sample_has_time(out_sample_has_time),
        .in_sample_time(out_sample_time),
        .in_sample_in_burst(out_sample_in_burst),
        .in_sample_countdown(out_sample_countdown));

    twbw_packer #(.HDR_XFERS(2)) packer(
        .clk ( SYS_clk ),
        .rst ( SYS_rst ),
        .in_tctrl(rx_packer_ctrl),
        .in_tdata(AXIS_RX_tdata_i0),
        .in_tlast(AXIS_RX_tlast_i0),
        .in_tvalid(AXIS_RX_tvalid_i0),
        .in_tready(AXIS_RX_tready_i0),
        .out_tdata(AXIS_RX_tdata_i1),
        .out_tkeep(AXIS_RX_tkeep_i1),
        .out_tlast(AXIS_RX_tlast_i1),
        .out_tvalid(AXIS_RX_tvalid_i1),
        .out_tready(AXIS_RX_tready_i1));

    stream_upsizer #(.IN_DATA_WIDTH(64))
    rf_rx_upsizer
    (
        .clk ( SYS_clk ),
        .rst ( SYS_rst ),
        .in_tdata(AXIS_RX_tdata_i1),
        .in_tkeep(AXIS_RX_tkeep_i1),
        .in_tlast(AXIS_RX_tlast_i1),
        .in_tvalid(AXIS_RX_tvalid_i1),
        .in_tready(AXIS_RX_tready_i1),
        .out_tdata(AXIS_RX_tdata_i2),
        .out_tkeep(AXIS_RX_tkeep_i2),
        .out_tlast(AXIS_RX_tlast_i2),
        .out_tvalid(AXIS_RX_tvalid_i2),
        .out_tready(AXIS_RX_tready_i2)
    );

    stream_combine_prio #(.DATA_WIDTH(128))
    RX_rf_combine(
        .clk(SYS_clk),
        .rst(SYS_rst),
        .in_tdata(AXIS_RX_tdata_i2),
        .in_tkeep(AXIS_RX_tkeep_i2),
        .in_tuser(1'b0), //RX ID
        .in_tlast(AXIS_RX_tlast_i2),
        .in_tvalid(AXIS_RX_tvalid_i2),
        .in_tready(AXIS_RX_tready_i2),

        .prio_tdata(AXIS_STAT_tdata),
        .prio_tkeep(16'hffff),
        .prio_tuser(1'b1), //STAT ID
        .prio_tlast(AXIS_STAT_tlast),
        .prio_tvalid(AXIS_STAT_tvalid),
        .prio_tready(AXIS_STAT_tready),

        .out_tdata(AXIS_RX_tdata),
        .out_tkeep(AXIS_RX_tkeep),
        .out_tuser(AXIS_RX_tuser),
        .out_tlast(AXIS_RX_tlast),
        .out_tvalid(AXIS_RX_tvalid),
        .out_tready(AXIS_RX_tready));

    //--------------------------------------------------------------------
    //-- cross domains between fclk and tx
    //--------------------------------------------------------------------
    wire [63:0] AXIS_TX_tdata_i0;
    wire [7:0] AXIS_TX_tkeep_i0;
    wire AXIS_TX_tlast_i0;
    wire AXIS_TX_tvalid_i0;
    wire AXIS_TX_tready_i0;

    wire [63:0] AXIS_TX_tdata_i1;
    wire AXIS_TX_tlast_i1;
    wire AXIS_TX_tvalid_i1;
    wire AXIS_TX_tready_i1;

    wire [63:0] in_sample_data;
    wire in_sample_valid;
    wire in_sample_ready;
    wire in_sample_end_burst;
    wire in_sample_wait_trigger;
    wire in_sample_has_time;
    wire [63:0] in_sample_time;

    wire [31:0] TX_data_a_int;
    wire [31:0] TX_data_b_int;

    twbw_tx_bus #(
        .DATA_WIDTH(64), .TIME_WIDTH(64),
        .DROP_TIMEOUT(100000)) //1 ms at 100 MHz sys clk
    u_tx_bus (
        //system clock domain
        .SYS_clk(SYS_clk),
        .SYS_rst(SYS_rst || tx_clear),

        //tx sample interface - sys clock domain
        .in_sample_data(in_sample_data),
        .in_sample_valid(in_sample_valid),
        .in_sample_ready(in_sample_ready),
        .in_sample_end_burst(in_sample_end_burst),
        .in_sample_wait_trigger(in_sample_wait_trigger),
        .in_sample_has_time(in_sample_has_time),
        .in_sample_time(in_sample_time),

        //status interface - sys clock domain
        .out_status_valid(out_status_valid),
        .out_status_ready(out_status_ready),
        .out_status_end_burst(out_status_end_burst),
        .out_status_underflow(out_status_underflow),
        .out_status_time_late(out_status_time_late),
        .out_status_has_time(out_status_has_time),
        .out_status_time(out_status_time),

        //tx clock domain interface (tx sample on each strobe cycle)
        .TX_clk(DATA_clk),
        .TX_rst(DATA_rst),
        .TX_stb(1'b1),
        .TX_active(DAC_tx_active),
        .TX_data({TX_data_b_int, TX_data_a_int}),

        //input events in rx clock domain
        .in_trigger(trigger_in),
        .in_time(time_now));

    assign AXIS_STAT_tdata = {
        42'b0, //meaning
        out_status_end_burst, out_status_underflow, out_status_time_late, out_status_has_time, out_status_valid, sequence_terror, sequence_tvalid, //86:80
        sequence_tdata, //79:64
        out_status_time}; //63:0
    assign AXIS_STAT_tlast = 1'b1;
    assign AXIS_STAT_tvalid = out_status_valid || sequence_tvalid;
    assign sequence_tready = AXIS_STAT_tready;
    assign out_status_ready = AXIS_STAT_tready;

    twbw_tx_deframer64 u_deframer (
        .clk(SYS_clk),
        .rst(SYS_rst),
        .sequence_tdata(sequence_tdata),
        .sequence_terror(sequence_terror),
        .sequence_tvalid(sequence_tvalid),
        .sequence_tready(sequence_tready),
        .in_tdata(AXIS_TX_tdata_i1),
        .in_tlast(AXIS_TX_tlast_i1),
        .in_tvalid(AXIS_TX_tvalid_i1),
        .in_tready(AXIS_TX_tready_i1),
        .out_sample_data(in_sample_data),
        .out_sample_valid(in_sample_valid),
        .out_sample_ready(in_sample_ready),
        .out_sample_end_burst(in_sample_end_burst),
        .out_sample_wait_trigger(in_sample_wait_trigger),
        .out_sample_has_time(in_sample_has_time),
        .out_sample_time(in_sample_time));

    stream_dnsizer #(.IN_DATA_WIDTH(128))
    rf_tx_dnsizer
    (
        .clk ( SYS_clk ),
        .rst ( SYS_rst ),
        .in_tdata(AXIS_TX_tdata),
        .in_tkeep(AXIS_TX_tkeep),
        .in_tlast(AXIS_TX_tlast),
        .in_tvalid(AXIS_TX_tvalid),
        .in_tready(AXIS_TX_tready),
        .out_tdata(AXIS_TX_tdata_i0),
        .out_tkeep(AXIS_TX_tkeep_i0),
        .out_tlast(AXIS_TX_tlast_i0),
        .out_tvalid(AXIS_TX_tvalid_i0),
        .out_tready(AXIS_TX_tready_i0)
    );

    twbw_unpacker #(.HDR_XFERS(2)) unpacker(
        .clk ( SYS_clk ),
        .rst ( SYS_rst ),
        .in_tctrl(tx_unpacker_ctrl),
        .in_tdata(AXIS_TX_tdata_i0),
        .in_tkeep(AXIS_TX_tkeep_i0),
        .in_tlast(AXIS_TX_tlast_i0),
        .in_tvalid(AXIS_TX_tvalid_i0),
        .in_tready(AXIS_TX_tready_i0),
        .out_tdata(AXIS_TX_tdata_i1),
        .out_tlast(AXIS_TX_tlast_i1),
        .out_tvalid(AXIS_TX_tvalid_i1),
        .out_tready(AXIS_TX_tready_i1));

    //--------------------------------------------------------------------
    //-- Time mgmt
    //--------------------------------------------------------------------
    wire [63:0] time_now_fdd;
    time_master RX_time_master(
        .SYS_clk(SYS_clk),
        .SYS_rst(SYS_rst),
        .SYS_time_in(SYS_time_next),
        .SYS_time_out(SYS_time_last_fdd),
        .SYS_time_write(SYS_time_write && SYS_time_event),
        .SYS_time_asap(SYS_time_asap && SYS_time_event),
        .SYS_time_read(SYS_time_read && SYS_time_event),
        .DATA_clk(DATA_clk),
        .DATA_rst(DATA_rst),
        .DATA_stb(1'b1),
        .DATA_trigger_in(trigger_in),
        .DATA_time_now(time_now_fdd));

    //--------------------------------------------------------------------
    //-- Tx sample ram to overlay DAC outputs when enabled
    //--------------------------------------------------------------------
    wire signed [31:0] TX_ram_data [0:1];
    wire DATA_tx_replay_enable;
    wire [15:0] DATA_tx_replay_start, DATA_tx_replay_stop;
    wire TX_ram_manual_enable;
    wire [$clog2(REPLAY_RAM_SIZE)-1:0] TX_ram_manual_addr;

    genvar ch;
    generate
    for (ch = 0; ch < 2 ; ch = ch + 1) begin
        tx_sample_ram #(.MEM_SIZE(REPLAY_RAM_SIZE), .DATA_WIDTH(32))
        tx_ram_cha(
            .SYS_clk(SYS_clk),
            .SYS_rst(SYS_rst),
            .wr_stb(tx_ram_write_stb[ch]),
            .wr_addr(tx_ram_write_addr[$clog2(REPLAY_RAM_SIZE)-1:0]),
            .wr_data(tx_ram_write_data[ch]),
            .DATA_clk(DATA_clk),
            .DATA_rst(DATA_rst),
            .replay_enable(DATA_tx_replay_enable),
            .replay_start(DATA_tx_replay_start[$clog2(REPLAY_RAM_SIZE)-1:0]),
            .replay_stop(DATA_tx_replay_stop[$clog2(REPLAY_RAM_SIZE)-1:0]),
            .manual_enable(TX_ram_manual_enable),
            .manual_addr(TX_ram_manual_addr),
            .out_data(TX_ram_data[ch]));
    end
    endgenerate

    xpm_cdc_array_single #(
        .WIDTH(1+16+16))
    tx_replay_enabled_xfer(
        .src_clk(SYS_clk),
        .src_in({tx_replay_enable, tx_replay_start, tx_replay_stop}),
        .dest_clk(DATA_clk),
        .dest_out({DATA_tx_replay_enable, DATA_tx_replay_start, DATA_tx_replay_stop}));

    wire signed [31:0] beacon_weight_data [0:1];
    wire [6:0] weight_max_addr;
    wire weighted_beacon; // = weight_max_addr == 0 ? 1'b0 : 1'b1;
    xpm_cdc_array_single #(
        .WIDTH(1+7))
    tx_ram_weight_xfer(
        .src_clk(SYS_clk),
        .src_in({SYS_weighted_beacon, SYS_weight_max_addr}),
        .dest_clk(DATA_clk),
        .dest_out({weighted_beacon, weight_max_addr}));

    wire [63:0] DATA_frame_now;
    reg [6:0] weight_ram_addr;
    wire [6:0] weight_ram_addr_next = weight_ram_addr + 1'b1;
    always @(posedge DATA_clk) begin
        if (DATA_rst || DATA_tdd_mode == 1'b0) weight_ram_addr <= '0;
        else if (trigger_out == 1'b1) weight_ram_addr <= (weight_ram_addr_next == weight_max_addr) ? '0 : weight_ram_addr_next;
    end
    
`ifndef CORR
    generate
    for (ch = 0; ch < 2 ; ch = ch + 1) begin
        array_ram #(.MEM_SIZE(128), .DATA_WIDTH(32))
        u_weight_buffer(
            .wr_clk(SYS_clk),
            .wr_rst(SYS_rst),
            .we(SYS_weight_stb[ch]),
            .wr_addr(SYS_weight_addr[6:0]),
            .wr_data(SYS_weight_din[ch]),
            .rd_clk(DATA_clk),
            .rd_rst(DATA_rst),
            .re(TX_ram_manual_enable),
            .rd_addr(weight_ram_addr),
            .rd_data(beacon_weight_data[ch]));
    end
    endgenerate
`endif

    reg [15:0] beacon_symbol;
    reg [15:0] beacon_stop;
    reg [15:0] beacon_start;
    xpm_cdc_array_single #(
        .WIDTH(16+16+16))
    beacon_bound_xfer(
        .src_clk(SYS_clk),
        .src_in({SYS_beacon_symbol, SYS_beacon_stop, SYS_beacon_start}),
        .dest_clk(DATA_clk),
        .dest_out({beacon_symbol, beacon_stop, beacon_start}));

    wire DATA_tdd_beacon;
    wire beacon_ram_read_enable;
    wire beacon_tx_zero; 
    reg [$clog2(BEACON_RAM_SIZE)-1:0] beacon_ram_addr;
    wire [31:0] beacon_ram_data;
`ifndef CORR
    array_ram #(.MEM_SIZE(BEACON_RAM_SIZE), .DATA_WIDTH(32))
    beacon_buffer(
        .wr_clk(SYS_clk),
        .wr_rst(SYS_rst),
        .we(SYS_beacon_stb),
        .wr_addr(SYS_beacon_addr[$clog2(BEACON_RAM_SIZE)-1:0]),
        .wr_data(SYS_beacon_din),
        .rd_clk(DATA_clk),
        .rd_rst(DATA_rst),
        .re(beacon_ram_read_enable),
        .rd_addr(beacon_ram_addr),
        .rd_data(beacon_ram_data));
`endif

    // Temporary implementation for Hadamard code! Correct implementation should be: TX_ram_weigt * TX_ram_data
    // Also the Weight RAM causes a delay for one cycle but given the prefix we're using, it
    // is fine for now
    wire signed [15:0] beacon_i = beacon_ram_data[15:0];
    wire signed [15:0] beacon_q = beacon_ram_data[31:16];
    wire [31:0] beacon_data_a_weighted = beacon_weight_data[0] == 0 ? '0 : ($signed(beacon_weight_data[0]) < 0 ? {-beacon_q, -beacon_i} : beacon_ram_data);
    wire [31:0] beacon_data_b_weighted = beacon_weight_data[1] == 0 ? '0 : ($signed(beacon_weight_data[1]) < 0 ? {-beacon_q, -beacon_i} : beacon_ram_data);
    
    wire [31:0] beacon_a_int = (weighted_beacon == 1'b1) ? beacon_data_a_weighted : beacon_ram_data;
    wire [31:0] beacon_b_int = (weighted_beacon == 1'b1) ? beacon_data_b_weighted : beacon_ram_data;

/*
ila_0 (
    .clk(DATA_clk),
    .probe0(TX_data_a[31:16]), 
    .probe1(TX_data_a[15: 0]),
    .probe2(TX_data_b[31:16]),
    .probe3(TX_data_b[15: 0]),
    .probe4(TX_ram_weight[0]), 
    .probe5(TX_ram_weight[1]),
    .probe6(DATA_frame_now[63:32]),
    .probe7(weight_ram_addr), 
    .probe8(weight_max_addr),
    .probe9(weighted_beacon),
    .probe10(DAC_tx_from_ram),
    .probe11(trigger_in)
);*/  
    //--------------------------------------------------------------------
    //-- frame schedule control
    //--------------------------------------------------------------------

    // additional TDD registers handling
    wire [31:0] DATA_tdd_conf;
    wire        DATA_tdd_stb;
    assign DATA_tdd_mode = DATA_tdd_conf[31];
    wire ext_sync_en = DATA_tdd_conf[30];
    wire wait_trigger = DATA_tdd_conf[29];
    wire alternating_ch_tx_ram = DATA_tdd_conf[28];
    wire [3:0] DATA_tdd_sched_count_max = DATA_tdd_conf[27:24];
    wire [15:0] DATA_tdd_symbols_per_frame;
    wire [15:0] DATA_tdd_samples_per_symbol;
    wire [31:0] DATA_tdd_frame_count_max;
    xpm_cdc_array_single #(.WIDTH(1+32+16+16+32))
    DATA_tdd_config_xfer(
        .src_clk(SYS_clk),
        .src_in({SYS_tdd_stb, SYS_tdd_conf, SYS_tdd_samples_per_symbol, SYS_tdd_symbols_per_frame, SYS_tdd_frame_count_max}),
        .dest_clk(DATA_clk),
        .dest_out({DATA_tdd_stb, DATA_tdd_conf, DATA_tdd_samples_per_symbol, DATA_tdd_symbols_per_frame, DATA_tdd_frame_count_max}));

    wire [1:0] DATA_tdd_phase;

    tdd_time_master #(.TIME_WIDTH(64))
    TDD_time_master(
        .SYS_clk(SYS_clk),
        .SYS_rst(SYS_rst),
        .SYS_sched_addr(SYS_sched_addr),
        .SYS_sched_din(SYS_sched_din),
        .SYS_sched_stb(SYS_sched_stb),
        .SYS_time_in(SYS_time_next),
        .SYS_time_out(SYS_time_last_tdd),
        .SYS_time_write(SYS_time_write && SYS_time_event),
        .SYS_time_asap(SYS_time_asap && SYS_time_event),
        .SYS_time_read(SYS_time_read && SYS_time_event),
        .DATA_clk(DATA_clk),
        .DATA_rst(DATA_rst | DATA_tdd_stb),
        .DATA_stb(DATA_tdd_mode),
        .ext_sync_en(ext_sync_en),
        .wait_trigger(wait_trigger),
        .samples_per_symbol(DATA_tdd_samples_per_symbol),
        .symbols_per_frame(DATA_tdd_symbols_per_frame),
        .sched_count_max(DATA_tdd_sched_count_max),
        .frame_count_max(DATA_tdd_frame_count_max),
        .DATA_trigger_in(trigger_in),
        .DATA_frame_end(trigger_out),
        .DATA_frame_now(DATA_frame_now),
        .DATA_tdd_en(/*open*/),
        .DATA_tdd_phase(DATA_tdd_phase),
        .DATA_rx_active(rx_tdd_active),
        .DATA_tx_active(/*open*/),
        .DATA_guard_active(/*open*/),
        .DATA_txram_active(txram_tdd_active),
        .DATA_rx_tlast(rx_tdd_eob));

    //pass frame time into the xx bus cores
    //for tx bus, this provides the time used to wait on
    //for the packets sent from the PC driver side
    //for rx bus, this provides the in-band timestamp
    //that is associated with the samples to be framed
    assign time_now = DATA_tdd_mode? DATA_frame_now : time_now_fdd;

    //assign tx ram control signals
    assign TX_ram_manual_enable = DATA_tdd_mode && txram_tdd_active;
    assign TX_ram_manual_addr   = DATA_frame_now[$clog2(REPLAY_RAM_SIZE)-1:0];
`ifdef CORR
    assign DATA_tdd_beacon = 0;
    assign beacon_ram_read_enable = 0;
    assign beacon_tx_zero = 0;
`else
    assign DATA_tdd_beacon = DATA_tdd_mode
			   && (DATA_tdd_phase == 2'b00)
			   && (beacon_symbol == DATA_frame_now[31:16]);
    assign beacon_ram_read_enable = DATA_tdd_beacon
				&& (beacon_start <= DATA_frame_now[15:0])
				&& (beacon_stop > DATA_frame_now[15:0]);

    assign beacon_tx_zero = DATA_tdd_beacon && !beacon_ram_read_enable;
`endif
    //assign beacon_ram_addr = beacon_ram_read_enable == 1 ? (DATA_frame_now[15:0] - beacon_start) : '0;
    wire [$clog2(BEACON_RAM_SIZE)-1:0] beacon_ram_addr_next = beacon_ram_addr + 1'b1;
    always @(posedge DATA_clk) begin
        if (DATA_rst || beacon_ram_read_enable == 1'b0) beacon_ram_addr <= '0;
        else beacon_ram_addr <= beacon_ram_addr_next;
    end
    
 
    wire [31:0] TX_ram_a_int = beacon_ram_read_enable ? beacon_a_int : (((alternating_ch_tx_ram && DATA_frame_now[16]) || beacon_tx_zero)  ? '0 : TX_ram_data[0]);
    wire [31:0] TX_ram_b_int = beacon_ram_read_enable ? beacon_b_int : (((alternating_ch_tx_ram && !DATA_frame_now[16]) || beacon_tx_zero) ? '0 : TX_ram_data[1]);

    assign DAC_tx_from_ram = DATA_tx_replay_enable | TX_ram_manual_enable | DATA_tdd_beacon;
    assign TX_data_a = DAC_tx_from_ram ? TX_ram_a_int : TX_data_a_int;
    assign TX_data_b = DAC_tx_from_ram ? TX_ram_b_int : TX_data_b_int;

endmodule //iris_rfcore

`default_nettype wire
