------------------------------------------------------------------------
-- TWBW Framer - ADC sample data framer with time and burst controls
--
-- Control port documentation:
--
-- Each control bus message consists of four 32-bit transactions.
-- The last 32-bit transaction in each message has a tlast asserted.
-- The message contains flags, a frame size, a burst size, and a time.
-- The tag is forwarded to the output port to help identify the message.
-- The burst and time fields are only applicable when the burst and time
-- flags are set respectively; but must always be present in the message.
--
-- Control bus format:
--   ctrl0[31] - time stamp flag
--   ctrl0[28] - finite burst flag
--   ctrl0[27] - continuous flag
--   ctrl0[26] - wait trigger flag
--   ctrl0[23:16] - identifying tag (forwarded to status port)
--   ctrl0[15:0] - frame size (number of transfers in a frame)
--   ctrl1[31:0] - burst size (only used with finite burst flag)
--   ctrl2[31:0] - 64 bit time high in clock ticks (used with time flag)
--   ctrl3[31:0] - 64 bit time low in clock ticks (used with time flag)
--
-- Output port documentation:
--
-- The output port contains framed sample data with 4 header lines.
-- The last transaction in each frame has a tlast asserted.
-- The output port data width may be larger than 32 bits,
-- but only the first 32 bits of each header line are used.
--
-- Output bus format:
--   out0[31] - time stamp flag indicates valid time
--   out0[30] - time error event occurred
--   out0[28] - finite burst flag (forwarded from control port)
--   out0[27] - continuous flag (forwarded from control port)
--   out0[26] - wait trigger flag (forwarded from control port)
--   out0[23:16] - identifying tag (forwarded from control port)
--   out0[15:0] - frame size (forwarded from control port)
--   out1[31:0] - current burst size counting down to zero
--   out2[31:0] - 64 bit time high in clock ticks (used with time flag)
--   out3[31:0] - 64 bit time low in clock ticks (used with time flag)
--
-- Copyright (c) 2015-2015 Lime Microsystems
-- Copyright (c) 2015-2015 Andrew "bunnie" Huang
-- SPDX-License-Identifier: Apache-2.0
-- http://www.apache.org/licenses/LICENSE-2.0
------------------------------------------------------------------------

library work;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity twbw_framer is
    generic(
        DATA_WIDTH : natural := 64;
        TIME_WIDTH : natural := 64);
    port(
        -- The ADC clock domain used for all interfaces.
        clk : in std_logic;

        -- synchronous reset
        rst : in std_logic;

        -- an external trigger used in trigger wait mode
        in_trigger : in std_logic;

        -- The current time in clock ticks.
        in_time : in unsigned(TIME_WIDTH-1 downto 0);

        -- Input ADC interface:
        -- There is no ready signal, a valid signal with no space => overflow.
        -- To allow for frame overhead, this bus cannot be valid every cycle.
        -- Many interfaces do not produce receive data at every clock cycle,
        -- If this is not the case, we recommend doubling the ADC data width.
        adc_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);
        adc_tvalid : in std_logic;
        adc_tready : out std_logic;

        -- Output stream interface:
        -- The tuser signal indicates metadata and not sample data.
        out_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);
        out_tuser : out std_logic_vector(0 downto 0);
        out_tlast : out std_logic;
        out_tvalid : out std_logic;
        out_tready : in std_logic;

        -- control bus interface
        ctrl_tdata : in std_logic_vector(31 downto 0);
        ctrl_tlast : in std_logic;
        ctrl_tvalid : in std_logic;
        ctrl_tready : out std_logic;

        -- Receive activity indicator based on internal state.
        -- Example: use this signal to drive external switches and LEDs.
        adc_active : out std_logic
    );
end entity twbw_framer;

architecture rtl of twbw_framer is

    -- state machine inspects the ADC from these signals
    signal adc_fifo_in_trigger : boolean;
    signal adc_fifo_in_ready : std_logic;
    signal adc_fifo_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal adc_fifo_out_time : unsigned(in_time'range);
    signal adc_fifo_out_trigger : boolean;
    signal adc_fifo_out_valid : std_logic;
    signal adc_fifo_out_ready : std_logic;

    -- state machine inspects the control messages from these signals
    signal ctrl_fifo_out_data : std_logic_vector(127 downto 0);
    signal ctrl_fifo_out_valid : std_logic;
    signal ctrl_fifo_out_ready : std_logic;

    -- state machine controls the outgoing framed bus signals
    signal framed_fifo_out_full : std_logic_vector(2+DATA_WIDTH-1 downto 0);
    signal framed_fifo_in_full : std_logic_vector(2+DATA_WIDTH-1 downto 0);
    signal framed_fifo_in_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal framed_fifo_in_hdr : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal framed_fifo_in_user : std_logic_vector(0 downto 0);
    signal framed_fifo_in_last : std_logic;
    signal framed_fifo_in_valid : std_logic;
    signal framed_fifo_in_ready : std_logic;

    -- internal adc active indicator
    signal adc_active_i : boolean;

    -- state machine enumerations
    type state_type is (
        STATE_CTRL_IDLE,
        STATE_CTRL_READ,
        STATE_WAIT_TIME,
        STATE_HDR0_OUT,
        STATE_HDR1_OUT,
        STATE_HDR2_OUT,
        STATE_HDR3_OUT,
        STATE_SAMPS_OUT);
    signal state : state_type;
    signal state_num : unsigned(7 downto 0);

    -- state variables set by the state machine
    signal time_flag : std_logic := '0';
    signal trigger_flag : std_logic := '0';
    signal burst_flag : std_logic := '0';
    signal continuous_flag : std_logic := '0';
    signal ctrl_tag : std_logic_vector(7 downto 0) := (others => '0');
    signal burst_size : unsigned(31 downto 0) := to_unsigned(0, 32);
    signal frame_size : unsigned(15 downto 0) := to_unsigned(0, 16);
    signal frame_count : unsigned(15 downto 0) := to_unsigned(0, 16);
    signal stream_time : unsigned(TIME_WIDTH-1 downto 0) := to_unsigned(0, TIME_WIDTH);
    signal frame_done : boolean := false;
    signal burst_done : boolean := false;
    signal overflow : std_logic := '0';
    signal time_error : std_logic := '0';
    signal first_packet : std_logic := '0';

begin
    assert (TIME_WIDTH <= 64) report "twbw_framer: time width too large" severity failure;
    assert (DATA_WIDTH >= 32) report "twbw_framer: data width too small" severity failure;

    state_num <= to_unsigned(state_type'pos(state), 8);

    --boolean condition tracking
    adc_active <= '1' when adc_active_i else '0';
    frame_done <= frame_count = to_unsigned(0, frame_count'length);
    burst_done <= burst_size = to_unsigned(0, burst_size'length);

    --------------------------------------------------------------------
    -- short fifo between adc out and state machine
    -- this fifo ensures a continuous streaming despite framing overhead
    --------------------------------------------------------------------
    adc_fifo: entity work.rx_sync_fifo
    port map (
        clk => clk,
        rst => rst,
        in_samps => adc_tdata,
        in_trigger => adc_fifo_in_trigger,
        in_time => in_time,
        in_valid => adc_tvalid,
        in_ready => adc_fifo_in_ready,
        out_samps => adc_fifo_out_data,
        out_trigger => adc_fifo_out_trigger,
        out_time => adc_fifo_out_time,
        out_valid => adc_fifo_out_valid,
        out_ready => adc_fifo_out_ready
    );

    adc_tready <= adc_fifo_in_ready;
    adc_fifo_out_ready <=
        framed_fifo_in_ready when state = STATE_SAMPS_OUT else --forwarding to fifo
        '0' when (state = STATE_WAIT_TIME and adc_fifo_out_trigger) else --plugged when triggered
        '0' when (framed_fifo_in_user = "1") else --plugged when header
        '1' when not adc_active_i else '0'; --always draining the fifo when inactive

    --hold previous cycle until input valid occurs
    --this makes the trigger condition sticky
    trigger_work: process (clk, trigger_flag, in_trigger, time_flag, in_time, stream_time)
        variable this_cycle_trigger : boolean;
        variable prev_cycle_trigger : boolean;
    begin

        --trigger conditions
        this_cycle_trigger := (first_packet = '1') and (
            (trigger_flag = '0' and time_flag = '0') or
            (trigger_flag = '1' and in_trigger = '1') or
            (time_flag = '1' and in_time = stream_time));

        --was this cycle or the previous triggered?
        adc_fifo_in_trigger <= this_cycle_trigger or prev_cycle_trigger;

        --sticky bit logic for triggered
        if (rising_edge(clk)) then
            if (adc_tvalid = '1') then
                prev_cycle_trigger := false;
            else
                prev_cycle_trigger := this_cycle_trigger;
            end if;
        end if;

    end process trigger_work;

    --------------------------------------------------------------------
    -- short fifo between ctrl bus and state machine
    -- this fifo changes size and gives storage space
    --------------------------------------------------------------------
    ctrl_fifo: entity work.ctrl_msg_fifo128
    port map (
        clk => clk,
        rst => rst,
        in_tdata => ctrl_tdata,
        in_tlast => ctrl_tlast,
        in_tvalid => ctrl_tvalid,
        in_tready => ctrl_tready,
        out_tdata => ctrl_fifo_out_data,
        out_tvalid => ctrl_fifo_out_valid,
        out_tready => ctrl_fifo_out_ready
    );

    ctrl_fifo_out_ready <= '1' when (state = STATE_CTRL_READ) else '0';

    --------------------------------------------------------------------
    -- fifo between state machine and output stream
    --------------------------------------------------------------------
    framed_fifo : entity work.StreamFifoSrl32
    port map (
        clk => clk,
        rst => rst,
        in_data => framed_fifo_in_full,
        in_valid => framed_fifo_in_valid,
        in_ready => framed_fifo_in_ready,
        out_data => framed_fifo_out_full,
        out_valid => out_tvalid,
        out_ready => out_tready
    );

    framed_fifo_in_data <= adc_fifo_out_data when (state = STATE_SAMPS_OUT) else framed_fifo_in_hdr;
    framed_fifo_in_user <= "1" when (
        state = STATE_HDR0_OUT or
        state = STATE_HDR1_OUT or
        state = STATE_HDR2_OUT or
        state = STATE_HDR3_OUT) else "0";

    framed_fifo_in_last <= '1' when (state = STATE_SAMPS_OUT and (
        time_error = '1' or overflow = '1' or -- an error event
        frame_done or -- the frame ended
        (burst_flag = '1' and burst_done) or -- burst mode and counted last sample in burst
        (continuous_flag = '1' and ctrl_fifo_out_valid = '1') -- continuous mode and a new control message has arrived
    )) else '0';

    framed_fifo_in_full <= framed_fifo_in_user & framed_fifo_in_last & framed_fifo_in_data;
    framed_fifo_in_valid <=
        adc_fifo_out_valid when (state = STATE_SAMPS_OUT) else framed_fifo_in_user(0);

    out_tdata <= framed_fifo_out_full(DATA_WIDTH-1 downto 0);
    out_tlast <= framed_fifo_out_full(DATA_WIDTH);
    out_tuser <= framed_fifo_out_full(DATA_WIDTH+1 downto DATA_WIDTH+1);

    --------------------------------------------------------------------
    -- framer state machine
    --------------------------------------------------------------------
    process (clk, state, first_packet, time_error, burst_flag, continuous_flag, trigger_flag, ctrl_tag, frame_size, burst_size, adc_fifo_out_time) begin

    framed_fifo_in_hdr <= (others => '0');
    if (state = STATE_HDR0_OUT) then
        framed_fifo_in_hdr(31) <= first_packet; --time is always valid on first packet
        framed_fifo_in_hdr(30) <= time_error;
        framed_fifo_in_hdr(28) <= burst_flag;
        framed_fifo_in_hdr(27) <= continuous_flag;
        framed_fifo_in_hdr(26) <= trigger_flag;
        framed_fifo_in_hdr(23 downto 16) <= ctrl_tag;
        framed_fifo_in_hdr(15 downto 0) <= std_logic_vector(frame_size);
    elsif (state = STATE_HDR1_OUT) then
        framed_fifo_in_hdr(31 downto 0) <= std_logic_vector(burst_size);
    elsif (state = STATE_HDR2_OUT) then
        framed_fifo_in_hdr(TIME_WIDTH-32-1 downto 0) <= std_logic_vector(stream_time(TIME_WIDTH-1 downto 32));
    elsif (state = STATE_HDR3_OUT) then
        framed_fifo_in_hdr(31 downto 0) <= std_logic_vector(stream_time(31 downto 0));
    end if;

    if (rising_edge(clk)) then
        if (rst = '1') then
            adc_active_i <= false;
            state <= STATE_CTRL_IDLE;
            time_flag <= '0';
            burst_flag <= '0';
            continuous_flag <= '0';
            trigger_flag <= '0';
            ctrl_tag <= (others => '0');
            burst_size <= to_unsigned(0, burst_size'length);
            frame_size <= to_unsigned(0, frame_size'length);
            frame_count <= to_unsigned(0, frame_count'length);
            stream_time <= to_unsigned(0, stream_time'length);
            overflow <= '0';
            time_error <= '0';
            first_packet <= '0';
        else case state is

        when STATE_CTRL_IDLE =>
            -- state idle always returns to inactive
            -- remain inactive until a command begins
            adc_active_i <= false;
            state <= STATE_CTRL_READ;
            overflow <= '0';
            time_error <= '0';

        when STATE_CTRL_READ =>
            if (ctrl_fifo_out_ready = '1' and ctrl_fifo_out_valid = '1') then
                --register all relevant fields from the control message
                time_flag <= ctrl_fifo_out_data(127);
                burst_flag <= ctrl_fifo_out_data(124);
                continuous_flag <= ctrl_fifo_out_data(123);
                trigger_flag <= ctrl_fifo_out_data(122);
                ctrl_tag <= ctrl_fifo_out_data(119 downto 112);
                frame_size <= unsigned(ctrl_fifo_out_data(111 downto 96));
                burst_size <= unsigned(ctrl_fifo_out_data(95 downto 64));
                stream_time <= unsigned(ctrl_fifo_out_data(TIME_WIDTH-1 downto 0));
                state <= STATE_WAIT_TIME;
                first_packet <= '1';
            end if;

        when STATE_WAIT_TIME =>
            frame_count <= frame_size;

            --wait for samples before making a decision
            if (adc_fifo_out_valid = '0') then
                --pass

            --continuation packet, leave state asap
            elsif (first_packet = '0') then
                adc_active_i <= true;
                state <= STATE_HDR0_OUT;
                stream_time <= adc_fifo_out_time;

            --no condition specified, go to header out
            elsif (time_flag = '0' and trigger_flag = '0' and adc_fifo_out_trigger) then
                adc_active_i <= true;
                state <= STATE_HDR0_OUT;
                stream_time <= adc_fifo_out_time;

            --trigger flag specified and triggered
            elsif (trigger_flag = '1' and adc_fifo_out_trigger) then
                adc_active_i <= true;
                state <= STATE_HDR0_OUT;
                stream_time <= adc_fifo_out_time;

            --time flag specified and time matched
            elsif (time_flag = '1' and adc_fifo_out_trigger) then
                adc_active_i <= true;
                state <= STATE_HDR0_OUT;
                stream_time <= adc_fifo_out_time;

            --the requested time has expired!
            elsif (time_flag = '1' and adc_fifo_out_time > stream_time) then
                time_error <= '1';
                state <= STATE_HDR0_OUT;
                stream_time <= adc_fifo_out_time;

            end if;

        when STATE_HDR0_OUT =>
            --wait for the output fifo to accept a transfer
            if (framed_fifo_in_valid = '1' and framed_fifo_in_ready = '1') then
                state <= STATE_HDR1_OUT;
            end if;

        when STATE_HDR1_OUT =>
            --wait for the output fifo to accept a transfer
            if (framed_fifo_in_valid = '1' and framed_fifo_in_ready = '1') then
                state <= STATE_HDR2_OUT;
            end if;

        when STATE_HDR2_OUT =>
            --wait for the output fifo to accept a transfer
            if (framed_fifo_in_valid = '1' and framed_fifo_in_ready = '1') then
                state <= STATE_HDR3_OUT;
            end if;

        when STATE_HDR3_OUT =>
            --wait for the output fifo to accept a transfer
            if (framed_fifo_in_valid = '1' and framed_fifo_in_ready = '1') then
                state <= STATE_SAMPS_OUT;
            end if;

        when STATE_SAMPS_OUT =>
            --overflow indicator is sticky, only cleared by idle state
            overflow <= overflow or (adc_fifo_out_valid and not adc_fifo_out_ready);

            --wait for the output fifo to accept a transfer
            if (framed_fifo_in_valid = '1' and framed_fifo_in_ready = '1') then
                frame_count <= frame_count - 1;
                burst_size <= burst_size - 1;
            end if;

            --end this state under the various conditions below
            if (framed_fifo_in_valid = '1' and framed_fifo_in_ready = '1' and framed_fifo_in_last = '1') then
                time_flag <= '0'; -- dont wait on time again
                trigger_flag <= '0'; -- dont wait on trigger again
                first_packet <= '0'; --header is out, clear
                if (overflow = '1' or time_error = '1') then
                    state <= STATE_CTRL_IDLE;
                elsif (burst_flag = '1' and burst_done) then
                    state <= STATE_CTRL_IDLE;
                elsif (continuous_flag = '1' and ctrl_fifo_out_valid = '1') then
                    state <= STATE_CTRL_READ;
                elsif (frame_done) then
                    state <= STATE_WAIT_TIME;
                else
                    state <= STATE_CTRL_IDLE;
                end if;
            end if;

        end case;
        end if;
    end if;

    end process;

end architecture rtl;
