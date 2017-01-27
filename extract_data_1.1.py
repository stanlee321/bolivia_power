

# Code for extract the information from the web
# with the <id> information into the bolivia_power_1.csv file


# input: bolivia_power_1.id.csv
# output 6x.npy  array file:
# <nodes_ids.lat,lon> <node.tags>
# <way.ids> <way.ref> <way.tags>
# ... 


# v. 1.1



#import pandas as pd
import numpy as np
import pandas as pd




# Data from Bolivia_power

path_to_csv_power_data = '/notebooks/Power/data/bolivia_power_1.csv'
df_bolivia_power= pd.read_csv(path_to_csv_power_data,delimiter=',',sep=',', error_bad_lines=False)
df_bolivia_power.columns = ['type','id','name_1','name_2','name_3','name_4']
df_bolivia_power.head()



# As array Type and id
df2_type = np.asarray(df_bolivia_power['type'])
df2_id = np.asarray(df_bolivia_power['id'])


# Return to Pandas DataFrame

data_frame_type = pd.DataFrame(df2_type)
data_frame_id = pd.DataFrame(df2_id)


print(len(df2_type))

# AS a unique DataFrame 

M = np.ones((len(df2_type),2))
data_frame = pd.DataFrame(M, columns=['type', 'id'])


data_frame['type'] = data_frame_type
data_frame['id'] = data_frame_id

data_frame.head()


## Extracting the data from the web

import urllib.request
from urllib.error import URLError, HTTPError

print("starting to download the files...")


# function fur Convert to pandasdataframe from str

##################FUR NODES #####################
import xml.etree.ElementTree as ET

#################################################
def iter_docs(author):

    author_attr = author.attrib
    for doc in author.iterfind('.//node'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict


def extract_data():
    node = []
    way = []
    relation= []
    r = 0
    

    for x in data_frame['type']:

        n = data_frame['id'][r]

        try:
            page = urllib.request.urlopen('http://api.openstreetmap.org/api/0.6/' + x + '/%d' %n)
            if x == 'node':
                node.append(page)
                print(".....node...: " + "%d" %n)
                print('http://api.openstreetmap.org/api/0.6/' + x + '/%d' %n)
                print(len(node), '/' , data_frame.shape[0])
                node[-1] = node[-1].read().decode()
                r +=1
                np.array(node).dump(open('/notebooks/Power/data/nodes.npy', 'wb'))
                print(node[-1])
            if x == 'way':
                way.append(page)
                print(".....way...: " + "%d" %n)
                print('http://api.openstreetmap.org/api/0.6/' + x + '/%d' %n)
                print(len(node)+len(way)+len(relation), '/' ,data_frame.shape[0])
                way[-1] = way[-1].read().decode()
                r +=1
                np.array(way).dump(open('/notebooks/Power/data/ways.npy', 'wb'))
                print(way[-1])

            if x == 'relation':
                relation.append(page)
                print(".....relation...: " + "%d" %n)
                print('http://api.openstreetmap.org/api/0.6/' + x + '/%d' %n)
                print(len(node)+len(way)+len(relation), '/' ,data_frame.shape[0])
                relation[-1] = relation[-1].read().decode()
                r +=1
                np.array(relation).dump(open('/notebooks/Power/data/relations.npy', 'wb'))
                print(relation[-1])

        except HTTPError:
            print('The server couldn\'t fulfill the request...node')
            r = r + 1 
            #print('Error code: ', e.code)
            if HTTPError == True:
                pass
        except URLError:

            r = r + 1
            print('We failed to reach a server...node')
            #print('Reason: ', e.reason)
            if URLError == True:
                pass
               


    print("sussessful ...!!!!!!!!!!!!")
    print("check your disk... :P")
    
    #return (node, way, relation)

extract_data()


print('finished node,way,relation')
print('saving list arrays  into disk....')
#node, ways, relations = extract_data()





"""""
xml_data = node[0]
etree = ET.fromstring(xml_data) #create an ElementTree object 
d = pd.DataFrame(list(iter_docs(etree)))


data_list=[] 					# create list for append every dataframe

for i in range(1,len(node)):

	xml_data = node[i]
	etree = ET.fromstring(xml_data) #create an ElementTree object 
	doc_df = pd.DataFrame(list(iter_docs(etree)))
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


xml_data = node[0]
etree = ET.fromstring(xml_data) #create an ElementTree object 
w = pd.DataFrame(list(iter_docs(etree)))


data_list_way=[] 					# create list for append every dataframe

for i in range(1,len(way)):

	xml_data = node[i]
	etree = ET.fromstring(xml_data) #create an ElementTree object 
	doc_df = pd.DataFrame(list(iter_docs_way(etree)))
	data_list.append(doc_df)
	w = w.append(data_list[-1],ignore_index=True)

w.head()

w.to_csv('/notebooks/Power/data/power_way.csv', sep=',', encoding='utf-8',index = False)

#########################################################################################
########################################################## FUR Relation ##################################################

def iter_docs_rel(author):

    author_attr = author.attrib
    for doc in author.iterfind('.//way'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict


xml_data = node[0]
etree = ET.fromstring(xml_data) #create an ElementTree object 
r = pd.DataFrame(list(iter_docs_rel(etree)))


data_list_way=[] 					# create list for append every dataframe

for i in range(1,len(relation)):

	xml_data = node[i]
	etree = ET.fromstring(xml_data) #create an ElementTree object 
	doc_df = pd.DataFrame(list(iter_docs_rel(etree)))
	data_list.append(doc_df)
	r = r.append(data_list[-1],ignore_index=True)

r.head()

r.to_csv('/notebooks/Power/data/power_rel.csv', sep=',', encoding='utf-8',index = False)	"""	