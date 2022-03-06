------------------------------------------------------------------------
-- Tx Replay - replay transmit waveforms from a RAM
--
-- The module accepts an axi input stream,
-- and produces an axi output stream.
-- And can operate in pass-through or replay mode
-- based on the configured enable signal.
------------------------------------------------------------------------

library work;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tx_replay is
    generic (
        DATA_WIDTH : natural := 64;
        -- How many samples can be stored into RAM
        FIFO_DEPTH : positive := 1024
    );
    port(
        -- The fifo clock
        clk : in std_logic;

        -- synchronous reset
        rst : in std_logic;

        -- replay enabled
        -- true to replay from RAM, false for pass-through
        replay_enabled : in std_logic;

        -- enable recording new samples into the RAM
        record_enabled : in std_logic;

        -- input bus interface
        in_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);
        in_tvalid : in std_logic;
        in_tready : out std_logic;

        -- output bus interface
        out_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);
        out_tvalid : out std_logic;
        out_tready : in std_logic
    );
end entity tx_replay;

architecture rtl of tx_replay is
    constant zeros : std_logic_vector(in_tdata'range) := (others => '0');

    signal has_data : std_logic;

    --ram signals
    signal We : std_logic;
    signal Re : std_logic;
    signal last_addr  : natural range 0 to FIFO_DEPTH-1;
    signal Wr_addr  : natural range 0 to FIFO_DEPTH-1;
    signal Rd_addr  : natural range 0 to FIFO_DEPTH-1;
    signal Rd_addr_next  : natural range 0 to FIFO_DEPTH-1;
    signal Wr_data : std_ulogic_vector(DATA_WIDTH-1 downto 0);
    signal Rd_data : std_ulogic_vector(DATA_WIDTH-1 downto 0);

    --internal stream signals
    signal in_tready_i : std_logic;
    signal out_tvalid_i : std_logic;
begin

    --output mux handles pass-through when replay is disabled
    out_tdata <= zeros when replay_enabled = '1' and has_data = '0' else
        std_logic_vector(Rd_data) when replay_enabled = '1' else in_tdata;
    out_tvalid_i <= has_data when replay_enabled = '1' else in_tvalid;
    in_tready_i <= out_tready; --maintain dac pacing by passing ready through

    in_tready <= in_tready_i;
    out_tvalid <= out_tvalid_i;

    --------------------------------------------------------------------
    -- write state machine
    --
    -- Write to the RAM starting from address 0 when record becomes enabled.
    -- And stop when record becomes disabled and store the stop address.
    --------------------------------------------------------------------
    process (clk, in_tvalid, record_enabled) begin
        if (rising_edge(clk)) then
            if (rst = '1') then
                Wr_addr <= 0;
                last_addr <= 0;
                has_data <= '0';
            elsif (replay_enabled = '0') then --held in reset when replay is off
                Wr_addr <= 0;
                last_addr <= 0;
                has_data <= '0';
            elsif (record_enabled = '0') then
                Wr_addr <= 0;
            elsif (We = '1') then
                Wr_addr <= Wr_addr + 1;
                last_addr <= Wr_addr;
                has_data <= '1';
            end if;
        end if;
    end process;

    We <= in_tvalid and in_tready_i and record_enabled;
    Wr_data <= std_ulogic_vector(in_tdata);

    --------------------------------------------------------------------
    -- read state machine
    --
    -- Read samples from the RAM in order from 0 to the stop address.
    -- When stop address is reached, then restart at address 0.
    --------------------------------------------------------------------
    process (clk, out_tready, replay_enabled) begin
        if (rising_edge(clk)) then
            if (rst = '1') then
                Rd_addr <= 0;
            elsif (replay_enabled = '0') then
                Rd_addr <= 0;
            elsif (replay_enabled = '1' and out_tvalid_i = '1' and out_tready = '1') then
                Rd_addr <= Rd_addr_next;
            end if;
            Re <= out_tready;
        end if;
    end process;

    Rd_addr_next <= 0 when (Rd_addr = last_addr) else Rd_addr + 1;

    --------------------------------------------------------------------
    -- replay ram holds samples
    --------------------------------------------------------------------
    replay_ram: entity work.dual_port_ram
    generic map (
        MEM_SIZE => FIFO_DEPTH,
        SYNC_READ => true
    )
    port map (
        Wr_clock => clk,
        We => We,
        Wr_addr => Wr_addr,
        Wr_data => Wr_data,

        Rd_clock => clk,
        Re => Re,
        Rd_addr => Rd_addr,
        Rd_data => Rd_data
    );

end architecture rtl;
