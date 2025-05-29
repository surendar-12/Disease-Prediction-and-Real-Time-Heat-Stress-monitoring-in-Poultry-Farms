import pickle
import urllib.request
import json
from time import sleep
while True:
  conn = urllib.request.urlopen("https://api.thingspeak.com/channels/565129/feeds.json?results=1")
  response = conn.read()
  print ("http status code=%s" % (conn.getcode()))
  data=json.loads(response)
  x=int(data['feeds'][0]['entry_id'])
  y=x
  conn.close()
  while x==y:
    conn = urllib.request.urlopen("https://api.thingspeak.com/channels/565129/feeds.json?results=1")
    response = conn.read()
    #print ("http status code=%s" % (conn.getcode()))
    data=json.loads(response)
    y=int(data['feeds'][0]['entry_id'])
    conn.close()

  conn = urllib.request.urlopen("https://api.thingspeak.com/channels/565129/feeds.json?results=1")
  response = conn.read()
  print ("http status code=%s" % (conn.getcode()))
  data=json.loads(response)
  c=float(data['feeds'][0]['field3'])
  d=float(data['feeds'][0]['field4'])
  e=float(data['feeds'][0]['field5'])
  conn.close()
  filename = 'knn_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  person_reports = [[c,d,e]]
  predicted = loaded_model.predict(person_reports)
  print("ANALYSING....")
  print(predicted[0])
  #dt="https://api.thingspeak.com/update?api_key=C4Y9L5WM6V3ZXB7U&field1="+str(a)+"&field2="+str(b)+"&field3="+str(c)+"&field4="+str(d)+"&field5="+str(e)+"&field6="+str(predicted[0])
  #conn = urllib.request.urlopen(dt)
  #response = conn.read()