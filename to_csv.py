import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# Read numpy array
node 		= np.load(open('/notebooks/Power/data/nodes.npy', 'rb'))
way  		= np.load(open('/notebooks/Power/data/ways.npy', 'rb'))
relation 	= np.load(open('/notebooks/Power/data/relations.npy', 'rb'))






#################################################
def iter_docs_node(author):

    author_attr = author.attrib
    for doc in author.iterfind('.//node'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict


xml_data = node[0]
etree = ET.fromstring(xml_data) #create an ElementTree object 
d = pd.DataFrame(list(iter_docs_node(etree)))


data_list=[] 					# create list for append every dataframe

for i in range(1,len(node)):

	xml_data = node[i]
	etree = ET.fromstring(xml_data) #create an ElementTree object 
	doc_df = pd.DataFrame(list(iter_docs_node(etree)))
	data_list.append(doc_df)
	d = d.append(data_list[-1],ignore_index=True)

d.head()

d.to_csv('/notebooks/Power/data/power_node.csv', sep=',', encoding='utf-8',index = False)

#########################################################################################
##############################################FUR WAYS#####################################################################

def iter_docs_way(author):

    author_attr = author.attrib
    for doc in author.iterfind('.//way'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict


xml_data = way[0]
etree = ET.fromstring(xml_data) #create an ElementTree object 
w = pd.DataFrame(list(iter_docs_way(etree)))


data_list_way=[] 					# create list for append every dataframe

for i in range(1,len(way)):

	xml_data = way[i]
	etree = ET.fromstring(xml_data) #create an ElementTree object 
	doc_df = pd.DataFrame(list(iter_docs_way(etree)))
	data_list.append(doc_df)
	w = w.append(data_list_way[-1],ignore_index=True)

w.head()

w.to_csv('/notebooks/Power/data/power_way.csv', sep=',', encoding='utf-8',index = False)

#########################################################################################
########################################################## FUR Relation ##################################################

def iter_docs_rel(author):

    author_attr = author.attrib
    for doc in author.iterfind('.//relation'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict


xml_data = relation[0]
etree = ET.fromstring(xml_data) #create an ElementTree object 
r = pd.DataFrame(list(iter_docs_rel(etree)))


data_list_relation = [] 					# create list for append every dataframe

for i in range(1,len(relation)):

	xml_data = relation[i]
	etree = ET.fromstring(xml_data) #create an ElementTree object 
	doc_df = pd.DataFrame(list(iter_docs_rel(etree)))
	data_list.append(doc_df)
	r = r.append(data_list_relation[-1],ignore_index=True)

r.head()

r.to_csv('/notebooks/Power/data/power_rel.csv', sep=',', encoding='utf-8',index = False)




"""
class To_csv:

	osm = [".//node",".//way",".//relation",".//tag"]

	
	


	data_list_relation = [] 

	def __init__(self, n_w_r , tpe, idn, osm_type):
		self.type = tpe
		self.id = idn
		self.osm = osm_type
		self.n_w_r = n_w_r
		To_csv.xml_data = self.n_w_r
		To_csv.etree = ET.fromstring(xml_data) #create an ElementTree object 
		To_csv.r = pd.DataFrame(list(iter_docs_rel(etree)))

	def iter_docs_rel(self, osm):
		author_attr = author.attrib
		for doc in author.iterfind(self.osm):
			doc_dict = author_attr.copy()
			doc_dict.update(doc.attrib)
			doc_dict['data'] = doc.text
			yield doc_dict


						# create list for append every dataframe

	for i in range(1,len(relation)):

		xml_data = relation[i]
		etree = ET.fromstring(xml_data) #create an ElementTree object 
		doc_df = pd.DataFrame(list(iter_docs_rel(etree)))
		data_list.append(doc_df)
		r = r.append(data_list_relation[-1],ignore_index=True)

	r.head()

	r.to_csv('/notebooks/Power/data/power_rel.csv', sep=',', encoding='utf-8',index = False)
	"""

