#!/usr/bin/env python3
#
# Generate batch config
# Copyright (c) 2019 Tomohiro Arakawa

import sys
import argparse
from argparse import RawDescriptionHelpFormatter
import json

# Scenarios
scenarios = {
    9988: {
        'name': 'SCE Qualification',
        'duration': 630,
        'traffic_scenario': 99880,
        'n_competitor_nodes': 10
    },
    7013: {
        'name': 'PE2 Alleys of Austin w/Points- 5 Team',
        'duration': 930,
        'traffic_scenario': 70130,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    7026: {
        'name':
        'PE2 Passive Incumbent w/Points - 5 Team',
        'duration':
        630,
        'traffic_scenario':
        70260,
        'n_competitor_nodes':
        10,
        'n_opponent_teams':
        2,
        'n_opponent_nodes':
        10,
        'incumbents': [{
            'gateway': True,
            'ImageName': 'incumbent-passive-v2-3',
            'ModemConfig': 'Node51Incumbent_7026.json'
        }]
    },
    7047: {
        'name': 'PE2 A Slice of Life w/Points - 5 Team',
        'duration': 630,
        'traffic_scenario': 70470,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    7065: {
        'name': 'Payline (2 Stage)',
        'duration': 360,
        'traffic_scenario': 70650,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    7074: {
        'name':
        'PE2 Jammers w/Points - 5 Team',
        'duration':
        870,
        'traffic_scenario':
        70740,
        'n_competitor_nodes':
        10,
        'n_opponent_teams':
        4,
        'n_opponent_nodes':
        10,
        'incumbents': [{
            'gateway': True,
            'ImageName': 'incumbent-jammer-v1-0',
            'ModemConfig': 'Node51Incumbent_7074.json'
        }, {
            'gateway': True,
            'ImageName': 'incumbent-jammer-v1-0',
            'ModemConfig': 'Node52Incumbent_7074.json'
        }, {
            'gateway': True,
            'ImageName': 'incumbent-jammer-v1-0',
            'ModemConfig': 'Node53Incumbent_7074.json'
        }]
    },
    7087: {
        'name': 'Wildfire w/Scores',
        'duration': 750,
        'traffic_scenario': 70870,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10,
    },
    8101: {
        'name': 'Trash Compactor',
        'duration': 750,
        'traffic_scenario': 81010,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10,
    },
    8204: {
        'name': 'Nowhere to Run Baby w/Scores',
        'duration': 570,
        'traffic_scenario': 82040,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10,
    },
    8301: {
        'name':
        'Active Incumbent  w/Scores',
        'duration':
        330,
        'traffic_scenario':
        83010,
        'n_competitor_nodes':
        10,
        'n_opponent_teams':
        4,
        'n_opponent_nodes':
        10,
        'incumbents': [{
            'gateway': True,
            'ImageName': 'incumbent-active-v1-1',
            'ModemConfig': 'Node51Incumbent_8301.json'
        }]
    },
    8401: {
        'name': 'PE2 Alleys of Austin Variant (10% Threshold)',
        'duration': 930,
        'traffic_scenario': 84010,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    8411: {
        'name': 'PE2 Alleys of Austin Variant (25% Threshold)',
        'duration': 930,
        'traffic_scenario': 84110,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    8502: {
        'name': 'San Juan: Threshold 10 percent w/Scores',
        'duration': 930,
        'traffic_scenario': 85020,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    8901: {
        'name': 'Temperature Rising w/Scores, 12-50% Threshold',
        'duration': 750,
        'traffic_scenario': 89010,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    8911: {
        'name': 'Temperature Rising w/Scores, 62-100% Threshold',
        'duration': 750,
        'traffic_scenario': 89110,
        'n_competitor_nodes': 10,
        'n_opponent_teams': 4,
        'n_opponent_nodes': 10
    },
    9971: {
        'name': 'Test Scenario #1 -50dbFS',
        'duration': 300,
        'traffic_scenario': 99711,
        'n_competitor_nodes': 10
    },
    9972: {
        'name': 'Test Scenario #2 -70dbFS',
        'duration': 300,
        'traffic_scenario': 99721,
        'n_competitor_nodes': 10
    },
    9973: {
        'name': 'Test Scenario #3 -90dbFS',
        'duration': 300,
        'traffic_scenario': 99731,
        'n_competitor_nodes': 10
    },
    9990: {
        'name':
        'SCE CIL Qualification',
        'duration':
        630,
        'traffic_scenario':
        99900,
        'n_competitor_nodes':
        10,
        'incumbents': [{
            'gateway': True,
            'ImageName': 'incumbent-passive-v2-2',
            'ModemConfig': 'Node51Incumbent_7025.json'
        }, {
            'gateway': False,
            'ImageName': 'sc2observer1-2',
            'ModemConfig': 'sc2observer_9989.json'
        }]
    },
    9991: {
        'name':
        'SCE CIL Qual 13db Pathloss',
        'duration':
        630,
        'traffic_scenario':
        99910,
        'n_competitor_nodes':
        10,
        'incumbents': [{
            'gateway': True,
            'ImageName': 'incumbent-passive-v2-2',
            'ModemConfig': 'Node51Incumbent_9991.json'
        }, {
            'gateway': False,
            'ImageName': 'sc2observer1-2',
            'ModemConfig': 'sc2observer_9989.json'
        }]
    }
}


# Parse args
class BGParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        self.print_help()
        sys.exit(2)


scenario_list_str = "\n\n[ Available RF scenarios ]\n"
for scenarioid, data in scenarios.items():
    scenario_list_str += '  {}: {}\n'.format(scenarioid, data['name'])
parser = BGParser(
    description='Generate batch config JSON file.',
    formatter_class=RawDescriptionHelpFormatter,
    epilog='Example: ./batchgen.py 9988 bare test-abcdef.tar.xz TestJob' +
    scenario_list_str)
parser.add_argument("RFScenarioID", type=int, help="RF Scenario ID")
parser.add_argument(
    "ImageFile",
    type=str,
    help="File name of the container image (without extension)")
parser.add_argument("ConfigFile",
                    type=str,
                    help="File name of the config file (with extension)")
parser.add_argument("BatchName", type=str, help="Name of batch job")
parser.add_argument("-s",
                    "--solo",
                    action="store_true",
                    help="Solo match mode (disable free play)")
parser.add_argument("-op",
                    "--opponents",
                    type=int,
                    choices=[1, 2, 3, 4],
                    help="Number of opponent teams")
args = parser.parse_args()

# Generate JSON
if args.RFScenarioID not in scenarios:
    sys.exit("Error: RF Scenario ID not found.")

scenario = scenarios[args.RFScenarioID]
freeplay_mode = True if (not args.solo) and (
    'n_opponent_teams' in scenario) else False
n_opponent_teams = 0
if freeplay_mode:
    if args.opponents is not None:
        n_opponent_teams = args.opponents
    else:
        n_opponent_teams = scenario['n_opponent_teams']
data = {}
data['BatchName'] = args.BatchName
data['Duration'] = scenario['duration']
data['RFScenario'] = args.RFScenarioID
data['TrafficScenario'] = scenario['traffic_scenario']
if freeplay_mode and 'lb_config_id' in scenario:
    data['LeaderboardConfigID'] = scenario['lb_config_id']
    data['LeaderboardConfigName'] = scenario['lb_config_name']

node_data = []
for i in range(1, scenario['n_competitor_nodes'] + 1):
    node_data_item = {}
    node_data_item['RFNode_ID'] = i
    node_data_item['ImageName'] = args.ImageFile
    node_data_item['ModemConfig'] = args.ConfigFile
    node_data_item['isGateway'] = True if i == 1 else False
    node_data_item['TrafficNode_ID'] = i
    node_data_item['node_type'] = "fp-self" if freeplay_mode else "competitor"
    node_data.append(node_data_item)

if freeplay_mode and 'n_opponent_teams' in scenario:
    for j in range(1, n_opponent_teams + 1):
        for k in range(1, scenario['n_opponent_nodes'] + 1):
            node_data_item = {}
            node_data_item['RFNode_ID'] = i + (
                j - 1) * scenario['n_opponent_nodes'] + k
            node_data_item['ImageName'] = "freeplay"
            node_data_item['ModemConfig'] = "freeplay.conf"
            node_data_item['isGateway'] = True if k == 1 else False
            node_data_item['TrafficNode_ID'] = i + (
                j - 1) * scenario['n_opponent_nodes'] + k
            node_data_item['node_type'] = "fp-opponent"
            node_data_item['opponent_team'] = j
            node_data.append(node_data_item)

if 'incumbents' in scenario:
    incumbents = scenario['incumbents']
    for i in range(len(incumbents)):
        node_data_item = {}
        node_data_item['RFNode_ID'] = 51 + i  # incumbent ID starts with 51
        node_data_item['ImageName'] = incumbents[i]['ImageName']
        node_data_item['ModemConfig'] = incumbents[i]['ModemConfig']
        node_data_item['isGateway'] = incumbents[i]['gateway']
        node_data_item['TrafficNode_ID'] = 51 + i
        node_data_item[
            'node_type'] = "fp-incumbent" if freeplay_mode else "competitor"
        node_data.append(node_data_item)

data['NodeData'] = node_data

# Print JSON to stdout
print(json.dumps(data, indent=2))
