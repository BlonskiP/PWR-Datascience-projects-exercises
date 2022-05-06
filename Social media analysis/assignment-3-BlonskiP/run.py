from DataCollector import collect_all
from NetworkCreator import Network_creator
from NetworkPloter import plot_Network


HASH_TAG="Sasin"
REGION="PL"#???
mentions = "@SasinJacek"

network = collect_all(HASH_TAG,'2020-05-01',False) #Gets all twits with tag and puts them in json
creator = Network_creator()
print('creating network')
network = creator.get_community_network(network,HASH_TAG)

