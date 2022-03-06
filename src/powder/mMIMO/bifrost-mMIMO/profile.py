"""
Resource provisioning script for mMIMO experiments on the POWDER test-bed for Project Odin

Instructions:
The Faros hub and pc are connected via a private 10Gbps link.
All Iris radios and the Faros hub should come up with address between 192.168.1.101 and 192.168.1.200.
These addresses are reachable by first logging in to "pc".

Repository: https://gitlab.flux.utah.edu/kwebb/faros-simple.git

Cloned Repository: https://github.com/bharathkeshavamurthy/bifrost-mMIMO.git
"""

# The imports
# noinspection PyUnresolvedReferences
import geni.urn as urn
import geni.rspec.pg as pg
import geni.portal as portal
# noinspection PyUnresolvedReferences
import geni.rspec.emulab as elab
# noinspection PyUnresolvedReferences
import geni.rspec.emulab.spectrum as spectrum

# Resource strings for the compute node(s)
COMPUTE_CLUSTER = 'd840'
COMPUTE_NODE_ID = 'pc19-meb'
COMPUTE_NODE_DEFAULT_OS_IMAGE = 'urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU16-64-STD'

# Resource strings for the mMIMO hardware
MASSIVE_MIMO_BS_RADIO_HW_TYPE = 'faros_sfp'
MASSIVE_MIMO_CLIENT_RADIO_HW_TYPE = 'iris030'

# Portal context
context = portal.Context()

# Spectrum parameters
# noinspection PyArgumentList
portal.context.defineStructParameter(
    'freq_ranges', 'Range', [], multiValue=True, min=1, multiValueTitle='Frequency ranges for over-the-air operation',
    members=[portal.Parameter('freq_min', 'Allowed Spectrum Lower Bound', portal.ParameterType.BANDWIDTH, 2560.0),
             portal.Parameter('freq_max', 'Allowed Spectrum Upper Bound', portal.ParameterType.BANDWIDTH, 2570.0)])

# Bind the context parameters
params = context.bindParameters()

# Verify freq_ranges members
for i, freq_range in enumerate(params.freq_ranges):
    if freq_range.freq_max - freq_range.freq_min < 1:
        portal.context.reportError(portal.ParameterError('Min and Max frequencies must be separated by at least 1 MHz',
                                                         ['freq_ranges[%d].freq_min' % i,
                                                          'freq_ranges[%d].freq_max' % i]))

# Verify the context parameters
context.verifyParameters()

# Create a request object to start building the RSpec
request = portal.context.makeRequestRSpec()

# Request a compute node from the specified cluster, name it "pc", load the default os image onto it, execute
#   Faros startup operations, and create & assign a 40Gbps interface for the LAN connection (remoting & data collection)
# faros_start.sh: DHCP daemon configuration; Git submodules update; and Install SoapySDR, Python libs, and C libs
# Note that there is a private 10Gbps link between the Faros aggregation hub and this PC
pc = request.RawPC('pc')
pc.hardware_type = COMPUTE_CLUSTER
pc.component_id = COMPUTE_NODE_ID
pc.disk_image = COMPUTE_NODE_DEFAULT_OS_IMAGE
pc.addService(pg.Execute(shell='sh', command='/usr/bin/sudo /local/repository/faros_start.sh'))
pc_if = pc.addInterface('pc_if', pg.IPv4Address('192.168.1.1', '255.255.255.0'))
pc_if.bandwidth = 4e7

# Request a mMIMO Base Station (BS), name it "mm", and create 3 interfaces to it for connections to the
#   Radio Heads (RHs) in the enclosure via LAN (for control & aggregation)
mm = request.RawPC('mm')
mm.hardware_type = MASSIVE_MIMO_BS_RADIO_HW_TYPE
mm_if1 = mm.addInterface('mm_if1')
mm_if2 = mm.addInterface('mm_if2')
mm_if3 = mm.addInterface('mm_if3')

# Request 'X' Iris clients with names "irx", and create interfaces for LAN connections (remoting & control)
ir1 = request.RawPC('ir1')
ir1.hardware_type = MASSIVE_MIMO_CLIENT_RADIO_HW_TYPE
ir1.component_id = 'iris03'
ir1_if = ir1.addInterface('ir1_if')

ir2 = request.RawPC('ir2')
ir2.hardware_type = MASSIVE_MIMO_CLIENT_RADIO_HW_TYPE
ir2.component_id = 'iris04'
ir2_if = ir2.addInterface('ir2_if')

# Connect the compute node ("pc"), mMIMO BS ("mm"), and the X Iris clients ("irx") to a LAN
lan = request.LAN('lan')
lan.vlan_tagging = False
lan.setNoBandwidthShaping()

lan.addInterface(pc_if)

lan.addInterface(mm_if1)
lan.addInterface(mm_if2)
lan.addInterface(mm_if3)

lan.addInterface(ir1_if)
lan.addInterface(ir2_if)

# Add frequency request(s)
for freq_range in params.freq_ranges:
    request.requestSpectrum(freq_range.freq_min, freq_range.freq_max, 100)

# Print the RSpec to the enclosing page
portal.context.printRequestRSpec()
