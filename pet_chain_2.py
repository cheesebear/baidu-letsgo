# -*- coding:utf-8 -*-
import numpy as np
import configparser as ConfigParser
import json
import random
import time
import base64
import pandas as pd
import datetime
import os
import shutil
import cv2
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from pyplotz.pyplotz import PyplotZ

# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import logging
LOG_FILENAME="log.txt"
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)

from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()

import water2
from sklearn.svm import SVC
from sklearn.externals import joblib

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题

from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

import requests
# import requests.packages.urllib3.util.ssl_
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL'

# from client import online_ocr


class PetChain():
	def __init__(self):
		self.degree_map = {
			0: "common",
			1: "rare",
			2: "excellence",
			3: "epic",
			4: "mythical",
		}

		self.buff= {
			#目前,上限，
			0: 1,
			1: 1,
			2: 1,
			3: 0,
			4: 0,
		}
		self.inventory = {
			#目前,下限,上限,
			0: [0,0,0],
			1: [5,0,0],
			2: [10,0,0],
			3: [7,130,1000],
			4: [1,5,1000],
		}
		self.sellprice = {
			#查看价格（低于将进一步请求详情）,基础价格
			0: [3000,499],
			1: [10000,898],
			2: [25000,2899],
			3: [800000,200000],
			4: [10000000,200000],
		}		
		self.buyprice = {
			#目前,上限
			0: [0,1],
			1: [0,300],
			2: [1001,100],
			3: [30001,100],
			4: [900000,100],
		}			  
		self.marketprice = {
			#
			0: {},
			1: {},
			2: {},
			3: {},
			4: {},
			5: {},
			6: {},
			7: {},
			8: {},
		}		
		self.degree_conf = {}
		self.interval = 1

		self.cookies = ''
		self.username = ''
		self.password = ''
		self.headers = {}
		self.clf=None
		self.lastValError=None
		self.lastValRequest=None
		self.getMarketErrorCount=0
		self.do_purchase=True
		# self.do_purchase=False
		self.petid_index=pd.DataFrame()

		self.get_headers()
		self.get_config()
		self.get_model()
		self.update_config()
		# self.read_petid_indexing()
		
		for path in [
			os.path.join(os.getcwd(),'data','captcha_dataset'),
			os.path.join(os.getcwd(),'record'),
			os.path.join(os.getcwd(),'temp'),
			os.path.join(os.getcwd(),'other'),
			os.path.join(os.getcwd(),'buysellrecord'),
			]:
			if not os.path.exists(path):
				os.makedirs(path)

	def read_petid_indexing(self):
		inputfilepath='record_letsgogold/sum.csv'
		self.petid_index=pd.read_csv(inputfilepath,  sep='\t')
		self.petid_index.set_index('petid')

	def update_config_depreciated(self):
		if self.do_purchase==False:
			return
		inputfilepath=os.path.join(os.getcwd(),'record','{}.csv'.format((datetime.datetime.now()-datetime.timedelta(hours=1)).strftime( '%Y-%m-%d_%H' )))
		print(inputfilepath)
		try:
			df=pd.read_csv(inputfilepath,  sep='\t')
		except Exception as e:
			print('读取上个小时记录失败')
		else:
			print('读取上个小时记录成功')
			amount_uplimit = {
				0: 0,
				1: 0,
				2: 0,
				3: 3001,
				4: 30001,
			}

			df.drop_duplicates(subset =['id',],inplace=True) 
			df.sort_values(by='amount',ascending=True,inplace=True)
			for i in range(5):
				if i in[0,]:
					continue
				df_tmp=df[df.rareDegree==i]
				df_tmp=df_tmp[df_tmp.amount>100 ]
				max_amount=df_tmp.amount.head(10).max()
				print(i,'\t',max_amount)
				if not np.isnan(max_amount):
					self.degree_conf[i] = min(max_amount,amount_uplimit[i])

				# out,bins_=pd.cut(np.array(df_tmp.amount),bins=range(100,1000,20),right=True, retbins=True,include_lowest=0)
				# print(out)
				# print(pd.value_counts(out),type(pd.value_counts(out)))
				# print('degree',i)
				# print(df_tmp.amount.value_counts())
		finally:
			print(self.degree_conf)
		# os._exit(0)

	def update_config(self):
		self.get_minprice_from_letsgo()

	def get_minprice_from_letsgo(self):
		for x in range(0,48):
			pass
			inputfilepath='record_letsgogold/v3_{}.csv'.format((datetime.datetime.now()-datetime.timedelta(hours=x)).strftime( '%Y-%m-%d_%H' ))
			try:
				print(os.path.getsize(inputfilepath))
				if os.path.getsize(inputfilepath) >15000000:
					df=pd.read_csv(inputfilepath,  sep='\t')
				else:
					continue
			except Exception as e:
				pass
			else:
				print(inputfilepath,'reading file succeeded')
				break
			finally:
				pass
		
		df.rareCounts=list(map(lambda x:int(x[0]),df.rareDegree))
		# print(df.rareCounts)
		for rareCounts in range(0,9):		
			print(rareCounts)
			self.marketprice[rareCounts]={}
			df_tmp=df[df.rareCounts==rareCounts]
			for i,label in enumerate(['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]):
				print(label,end='\t')
				df_tmp[label]=df_tmp[label].fillna('other')
				values=list(set(list(df_tmp[label])))
				self.marketprice[rareCounts][label]={}
				for value in values:
					if rareCounts<=5:
						percentile_loc=2
					else:
						percentile_loc=2
					# if rareCounts==7:
					# 	print(list(df_tmp[df_tmp[label]==value]['amount']))
					self.marketprice[rareCounts][label][value]=np.percentile(list(df_tmp[df_tmp[label]==value]['amount']),percentile_loc)
					print(value,len(df_tmp[df_tmp[label]==value].index),self.marketprice[rareCounts][label][value],end='\t')
				print('\n')		
		print(self.marketprice)	

	def get_model(self):
		self.clf = joblib.load('data/letter.pkl')

	def get_config(self):
		config = ConfigParser.ConfigParser()
		config.read("config.ini")

		for i in range(5):
			try:
				amount = config.getfloat("Pet-Chain", self.degree_map.get(i))
			except Exception as e:
				amount = 100
			self.degree_conf[i] = amount

		self.interval = config.getfloat("Pet-Chain", "interval")


	def get_headers(self):
		with open("data/headers.txt") as f:
			lines = f.readlines()
			for line in lines:
				splited = line.strip().split(":")
				key = splited[0].strip()
				value = ":".join(splited[1:]).strip()
				self.headers[key] = value

	def get_market(self):
		global get_market_time
		get_market_time=time.time()
		try:
			data = {
				"appId": 1,
				"lastAmount": None,
				"lastRareDegree": None,
				"pageNo": 1,
				"petIds": [],
				"pageSize": 20,
				# "querySortType": "AMOUNT_ASC",
				# "pageSize": 20,				
				"querySortType": "CREATETIME_DESC",
				# "querySortType": "CREATETIME_DESC",
				"requestId": 1517968317687,
				"tpl": "",
			}

			self.headers['Referer'] = "https://pet-chain.baidu.com/chain/dogMarket?t={}&appId=1".format(int(time.time() * 1000))

			page = requests.post("https://pet-chain.baidu.com/data/market/queryPetsOnSale", headers=self.headers,
								 data=json.dumps(data), verify=False)
			
			# page = requests.post("http://121.42.239.33:8081/dogMarket/marketList", headers=self.headers,
			# 					 data=json.dumps(data), verify=False)


			# print(page,type(page))
			if page.json().get(u"errorMsg") == u"success":
				# print "[->] purchase"
				# print ".",
				pets = page.json().get(u"data").get("petsOnSale")


				print(len(pets),datetime.datetime.now().strftime( '%H:%M:%S' ))
				
				df=pd.DataFrame(pets)
				# df=df[df.rareDegree>0]
				# for pet in pets:
				
				for pet in df.sort_values(by=['rareDegree'],ascending=[0],inplace=False).to_dict(orient='records'):
					# print(pet)
					if self.do_purchase== True:
						# if (self.lastValRequest is not None) and time.time()-self.lastValRequest<5:
						# 	# print('限制验证码频次，跳过。')
						# 	continue
						if self.lastValError==None:
							self.purchase(pet)
						elif time.time()-self.lastValError>10:
							self.lastValError=None


				# pets_back=df.to_dict(orient='records')
				# print(pets_back)

				df['time']=datetime.datetime.now()
				
				filename=os.path.join(os.getcwd(),'record','{}.csv'.format(datetime.datetime.now().strftime( '%Y-%m-%d_%H' )))
				if os.path.exists(filename):
					df.to_csv(filename, sep='\t', encoding='utf-8',index=False,mode='a',header=False)
					# print(filename,'added')
				else:
					df.to_csv(filename, sep='\t', encoding='utf-8',index=False,mode='a',header=True)
					print(filename,'created')
					# self.update_config()
			self.getMarketErrorCount=0
		except Exception as e:
			print ('failed')
			self.getMarketErrorCount+=1
			pass
			# raise

	def purchase(self, pet):
		global get_market_time		
		try:
			pet_id = pet.get(u"petId")
			pet_amount = pet.get(u"amount")
			pet_degree = pet.get(u"rareDegree")
			pet_validCode = pet.get(u"validCode")
			
			if pet_degree in [0,1,2]:
				# print('品级太低，就不折腾了')
				return
			if self.inventory[pet_degree][0]>=self.inventory[pet_degree][2]:
				return
			print('pet_degree',pet_degree,'pet_amount',pet_amount,'pet_id',pet_id)

			##request for detail
			# try:
			# 	attributes_dict=self.petid_index.loc[pet_id,:].to_dict(orient='records')
			# 	print(attributes_dict)
			# 	rareCounts=attributes_dict['rareCounts']
			# 	print('read from cached history')
			# except Exception as e:
			try:
				querypet_data = {
					"appId": 1,
					"petId": pet_id,
					"requestId": int(time.time() * 1000),
					"tpl": ""
				}

				self.headers[
					'Referer'] = "https://pet-chain.baidu.com/chain/detail?channel=market&petId={}&appId=1&validCode={}".format(
					pet_id, pet_validCode)
				# print(querypet_data)
				page = requests.post("https://pet-chain.baidu.com/data/pet/queryPetById", headers=self.headers,
									 data=json.dumps(querypet_data), verify=False)
				# print(page)
				resp = page.json()
				attributes=resp.get(u"data").get(u"attributes")
				canbesold,rareCounts,attributes_dict,shouldbuy,sellratio,buyratio=self.analyze_detail(attributes)
				# print('request for detail completed')
			except Exception as e:
				print( e)
			print('end requesting for pet detail',round(0-get_market_time+time.time(),2),'s')

			
			if rareCounts<5:
				return
			maxprice=0
			maxvalue=None


			for i,label in enumerate(['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]):
				value=attributes_dict[label]
				try:					
					# print(label,value,self.marketprice[rareCounts][label][value])
					if self.marketprice[rareCounts][label][value]>maxprice:
						maxprice=self.marketprice[rareCounts][label][value]
						maxvalue=value
				except Exception as e:
					pass
			# if attributes_dict[u'体型']!=u'天使' and  attributes_dict[u'眼睛']!=u'白眉斗眼' and attributes_dict[u'嘴巴']!=u'樱桃':
			# 	return
			if attributes_dict[u'体型']==u'菠萝头':
				# print('菠萝头',end=' ')	
				maxprice*=0.7			
			if attributes_dict[u'眼睛']==u'小头晕':
				# print('小头晕',end=' ')				
				maxprice*=0.9
			if attributes_dict[u'体型']==u'天使' and  attributes_dict[u'眼睛']==u'白眉斗眼' and attributes_dict[u'嘴巴']==u'樱桃':
				maxprice*=6
			elif attributes_dict[u'体型']==u'天使' and  attributes_dict[u'眼睛']==u'白眉斗眼':
				maxprice*=3


			print(maxvalue,maxprice)
			if float(pet_amount)>0.7*maxprice:
				return



			data = {
				"appId": 1,
				"petId": pet_id,
				"captcha": "",
				"seed": 0,
				"requestId": int(time.time() * 1000),
				"tpl": "",
				"amount": "{}".format(pet_amount),
				"validCode": pet_validCode
			}


			
				
			if (self.lastValRequest is not None) and time.time()-self.lastValRequest<1:
				print('限制验证码频次，跳过')
				return				
			self.lastValRequest=time.time()

			print('start requesting for validation code',round(0-get_market_time+time.time(),2),'s')
			captcha, seed ,img= self.get_captcha()
			# time.sleep(5)
			assert captcha and seed, ValueError("验证码为空")
			print('end recognizing validation code',round(0-get_market_time+time.time(),2),'s')

			jump_data = {
				"appId": 1,
				"requestId": int(time.time() * 1000),
				"tpl": ""
			}

			page = requests.post("https://pet-chain.baidu.com/data/market/shouldJump2JianDan", headers=self.headers,
								 data=json.dumps(jump_data), verify=False, timeout=2)

			data['captcha'] = captcha
			data['seed'] = seed
			self.headers['Referer'] = "https://pet-chain.baidu.com/chain/detail?channel=market&petId={}&appId=1&validCode={}".format(pet_id, pet_validCode)
			page = requests.post("https://pet-chain.baidu.com/data/txn/create", headers=self.headers,
								 data=json.dumps(data), verify=False, timeout=2)
			resp = page.json()
			if resp.get(u"errorMsg") != u"验证码错误":
				# cv2.imwrite("data/captcha_dataset/%s.jpg" % captcha, image)
				self.lastValError=None
				dest_filepath=os.path.join(os.getcwd(),'data','captcha_dataset','{}.jpg' .format(captcha))
				# source_filepath=os.path.join(os.getcwd(),'data','captcha.jpg')
				# shutil.copy(source_filepath, dest_filepath)
				
				with open(dest_filepath, 'wb') as fp:
					fp.write(base64.b64decode(img))
					fp.close()					

				# cv2.imwrite(filename, img_np)					
				
				print ("Get one captcha sample")
			else:
				self.lastValError=time.time()
				# cv2.imwrite("data/captcha_dataset/neg_sample/%s.jpg" % str(time.time()).replace('.', '_'), image)
				
				dest_filepath=os.path.join(os.getcwd(),'data','captcha_wrongmark','{}.jpg' .format(captcha))
				with open(dest_filepath, 'wb') as fp:
					fp.write(base64.b64decode(img))
					fp.close()							

				print ("Get one negative sample")
			if resp.get(u"errorMsg") == u"success":

				log_msg='\t'+pet_id+'\t'+pet_amount+'\t'+pet_degree+'\t'+'bought'+'\t'+datetime.datetime.now()
				logging.info(log_msg)
			print (json.dumps(resp, ensure_ascii=False))
		except Exception as e:
			print(e)
			# raise




	def get_captcha(self):
		
		seed = -1
		captcha = -1
		try:
			data = {
				"requestId": int(time.time() * 1000),
				"appId": 1,
				"tpl": ""
			}
			page = requests.post("https://pet-chain.baidu.com/data/captcha/gen", data=json.dumps(data),
								 headers=self.headers, verify=False)
			resp = page.json()
			# print(resp.get(u"errorMsg"))
			if resp.get(u"errorMsg") == u"success":
				seed = resp.get(u"data").get(u"seed")
				img = resp.get(u"data").get(u"img")
				# print(img)
				# with open('data/captcha.jpg', 'wb') as fp:
				#	 fp.write(base64.b64decode(img))
				#	 fp.close()
				# im = Image.open("data/captcha.jpg")
				im =Image.open(BytesIO(base64.b64decode(img)))
				# im.show()
				# captcha = input("enter captcha:")
				# if captcha=='stop!':
				#	 self.do_purchase=False
				# im.close()
				t1=time.time()
				captcha=water2.reg_veri_code(im,self.clf)
				t2=time.time()
				print('takes',round(t2-t1,2),'seconds')
				# input()
		except Exception as e:
			print (e)
			# raise
		
		return captcha, seed,img
	
	def format_cookie(self, cookies):
		self.cookies = ''
		for cookie in cookies:
			self.cookies += cookie.get(u"name") + u"=" + cookie.get(u"value") + ";"
		self.headers = {
			'Accept': 'application/json',
			'Accept-Encoding': 'gzip, deflate, br',
			'Accept-Language': 'en-US,en;q=0.9',
			'Connection': 'keep-alive',
			'Content-Type': 'application/json',
			'Cookie': self.cookies,
			'Host': 'pet-chain.baidu.com',
			'Origin': 'https://pet-chain.baidu.com',
			'Referer': 'https://pet-chain.baidu.com/chain/dogMarket?t=1517829948427',
			'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36',
		}

		with open("data/headers.txt", "w") as f:
			for key, value in self.headers.items():
				f.write("{}:{}\n".format(key, value))


	def get_and_record_inventory(self):
		mypets ,inventory_counts=self.get_inventory()
		# print(inventory_counts,type(inventory_counts))
		for i in inventory_counts.index:
			self.inventory[i][0]=inventory_counts[i]
		print(self.inventory)

	def run(self):

		self.get_inventory()
		
		scheduler.add_job( self.get_inventory,'interval', minutes=1, id='my_job_id', misfire_grace_time=None)
		scheduler.add_job( self.update_config,'cron', minute=1, id='my_job_id2', misfire_grace_time=None)
		scheduler.start()
		
		while True:
			# print(time.time())
			if self.getMarketErrorCount>4:
				time.sleep(60*20)
			self.get_market()
			time.sleep(float(random.randint(1,2))/1.0)
			# time.sleep(float(random.uniform(2, 5))/1.0)

			# time.sleep(self.interval)
	
	def bat_get_veri_code(self):
		seed = -1
		captcha = -1
		while True:
			try:
				data = {
					"requestId": int(time.time() * 1000),
					"appId": 1,
					"tpl": ""
				}
				page = requests.post("https://pet-chain.baidu.com/data/captcha/gen", data=json.dumps(data),
									 headers=self.headers, verify=False)
				resp = page.json()
				print(resp.get(u"errorMsg"))
				if resp.get(u"errorMsg") == u"success":
					seed = resp.get(u"data").get(u"seed")
					img = resp.get(u"data").get(u"img")
					# print(img)
					# with open('data/captcha.jpg', 'wb') as fp:
					#	 fp.write(base64.b64decode(img))
					#	 fp.close()
					# im = Image.open("data/captcha.jpg")
					im =Image.open(BytesIO(base64.b64decode(img)))
					# im.show()
					# captcha = input("enter captcha:")
					# if captcha=='stop!':
					#	 self.do_purchase=False
					# im.close()
					captcha=water2.reg_veri_code(im,self.clf)
					# print(captcha)
					# input()
			except Exception as e:
				print (e)
				# raise
			time.sleep(float(random.randint(1, 4))/1.0) 
		# return captcha, seed,img

	def bat_get_veri_code2(self):
		while True:
			try:
				data = {
					"requestId": int(time.time() * 1000),
					"appId": 1,
					"tpl": ""
				}
				page = requests.post("https://pet-chain.baidu.com/data/captcha/gen", data=json.dumps(data),
									 headers=self.headers, verify=False)
				resp = page.json()
				if resp.get(u"errorMsg") == u"success":
					seed = resp.get(u"data").get(u"seed")
					img = resp.get(u"data").get(u"img")
					# print(img)
					image=base64.b64decode(img)
					image=np.array(image)
					nparr = np.fromstring(image,np.uint8)  
					img_np = cv2.imdecode(nparr,cv2.IMREAD_COLOR)  
					filename=os.path.join(os.getcwd(),'data','captcha_dataset2','{}.jpg' .format(time.time()))
					# cv2.imshow('image',img_np)
					# input()
					cv2.imwrite(filename, img_np)
			except Exception as e:
				print(e)	 
			time.sleep(float(random.randint(1, 4))/1.0)   

	def get_inventory(self):
		mypets=[]
		inventory_counts=[]
		
		page_no=1
		while True:
			try:   
				data = {
					"requestId": int(time.time() * 1000),
					"appId": 1,
					"tpl": "",
					"pageNo":page_no,
					"pageSize":20,
					"pageTotal":-1,
					"timeStamp":"",
					"nounce":"",
					"token":""
				}
				page = requests.post("https://pet-chain.baidu.com/data/user/pet/list", data=json.dumps(data),
									 headers=self.headers, verify=False)
				resp = page.json()
				if resp.get(u"errorMsg") == u"success":
					mypets_onepage = resp.get(u"data").get(u"dataList")
					if mypets_onepage==[] or page_no>10:
						break
					else:
						mypets+=mypets_onepage

					# print(inventory_counts)			
			except Exception as e:
				print(e)  
			page_no+=1

		try:
			df=pd.DataFrame(mypets)
			df.petId=list(map(lambda x:str(x),df.petId))
			df.to_csv('data/inventory.csv', sep='\t', encoding='utf-8',index=False)
			inventory_counts=df.rareDegree.value_counts()
			for i in inventory_counts.index:
				self.inventory[i][0]=inventory_counts[i]
			print(self.inventory)		
		except Exception as e:
			# raise e
			print (e)



		return mypets ,inventory_counts

	def modify_inventory(self):
		try:
			pass
			mypets ,inventory_counts=self.get_inventory()
			# mypets=mypets[::-1]  #reverse
			for pet in mypets:
				self.withdraw_dog(pet)
				self.sell_dog(pet)
		except Exception as e:
			# raise e
			pass

			
	def withdraw_dog(self,pet):
		# print(pet,type(pet))
		shelfStatus=pet.get(u"shelfStatus")
		chainStatus=pet.get(u"chainStatus")
		rareDegree=pet.get(u"rareDegree")
		amount=pet.get(u"amount")
		petId=pet.get(u"petId")
		id=pet.get(u"id")	
		if shelfStatus==0:
			return 
		try:
			# data = {
			#	 "petId":petId,
			#	 "requestId": int(time.time() * 1000),
			#	 "appId":1,
			#	 "tpl":"",
			#	 "timeStamp":"",
			#	 "nounce":"",
			#	 "token":""				
			# }
			# page = requests.post("https://pet-chain.baidu.com/data/pet/queryPetByIdWithAuth", data=json.dumps(data),
			#					  headers=self.headers, verify=False)
			# # resp = page.json()

			data = {
				"petId":petId,
				"requestId": int(time.time() * 1000),
				"appId":1,
				"tpl":"",
				"timeStamp":"",
				"nounce":"",
				"token":""	  


			}
			page = requests.post("https://pet-chain.baidu.com/data/market/unsalePet", data=json.dumps(data),
								 headers=self.headers, verify=False)
			resp = page.json()			
			if resp.get(u"errorMsg") == u"success":
				print('withdraw from shelf:','rareDegree',rareDegree,"petId",petId)
				pet['shelfStatus']=0
			else:
				print("petId",petId,resp.get(u"errorMsg"),resp.get(u"errorNo"))				
		except Exception as e:
			print(e) 
			raise   

	def analyze_detail(self,attributes):
		rareCounts=0
		canbesold=True
		shouldbuy=False
		sellratio=1
		buyratio=1
		attributes_dict={}
		for attribute in attributes:
			name=attribute.get(u"name")
			rareDegree=attribute.get(u"rareDegree")
			value=attribute.get(u"value")
			attributes_dict[name]=value
			if rareDegree==u'稀有':
				rareCounts+=1
		# print(attributes_dict)
		# if rareCounts==5 :
		# 	print('五稀',end=' ')
		# 	canbesold=False
		if attributes_dict[u'体型']==u'天使' :
			print('天使',end=' ')			
			shouldbuy=True
		else:
			sellratio*=0.35

		if attributes_dict[u'眼睛']==u'白眉斗眼':
			print('白眉斗眼',end=' ')			
			shouldbuy=True
		else:
			sellratio*=0.45
		if attributes_dict[u'嘴巴']==u'樱桃':
			print('樱桃',end=' ')			
			# shouldbuy=True
		else:
			sellratio*=0.45				
		if attributes_dict[u'体型']==u'菠萝头':
			print('菠萝头',end=' ')	
			shouldbuy=False			
		if attributes_dict[u'眼睛']==u'小头晕':
			print('小头晕',end=' ')	
			shouldbuy=False				
		if rareCounts==4:
			sellratio*=0.5					
			
		print('rareCounts',rareCounts)
		# print('sellratio',round(sellratio,2))

		return canbesold,rareCounts,attributes_dict,shouldbuy,sellratio,buyratio
	
	def func_canbesold(self,attributes):
		rareCounts=0
		canbesold=True
		shouldbuy=False
		sellratio=1
		buyratio=1
		attributes_dict={}
		for attribute in attributes:
			name=attribute.get(u"name")
			rareDegree=attribute.get(u"rareDegree")
			value=attribute.get(u"value")
			attributes_dict[name]=value
			if rareDegree==u'稀有':
				rareCounts+=1
		# print(attributes_dict)
		# if rareCounts==5 :
		# 	print('五稀',end=' ')
		# 	canbesold=False
		if attributes_dict[u'体型']==u'天使' :
			print('天使',end=' ')			
			canbesold=False
		if attributes_dict[u'眼睛']==u'白眉斗眼':
			print('白眉斗眼',end=' ')			
			canbesold=False		
		if attributes_dict[u'嘴巴']==u'大胡子':
			print('大胡子',end=' ')			
			canbesold=False					
		if attributes_dict[u'肚皮色']==u'异光蓝':
			print('异光蓝',end=' ')			
			canbesold=False					


		return canbesold,rareCounts,attributes_dict




	def sell_dog(self,pet):
		shelfStatus=pet.get(u"shelfStatus")
		chainStatus=pet.get(u"chainStatus")
		rareDegree=pet.get(u"rareDegree")
		amount=pet.get(u"amount")
		petId=pet.get(u"petId")
		id=pet.get(u"id")	
		if shelfStatus==1:
			return 
		# if rareDegree>3 and petId!='1894434276688471891':
		# 	print('高品级狗,不卖',petId)			
		# 	return	
		if rareDegree in[]:
			return
		if self.inventory[rareDegree][0]<self.inventory[rareDegree][1]:
			return
		# print("petId",petId)
		try:
			data = {
				"petId":petId,
				"requestId": int(time.time() * 1000),
				"appId":1,
				"tpl":"",
				"timeStamp":"",
				"nounce":"",
				"token":""				
			}
			page = requests.post("https://pet-chain.baidu.com/data/pet/queryPetByIdWithAuth", data=json.dumps(data),
								 headers=self.headers, verify=False)
			resp = page.json()
			attributes=resp.get(u"data").get(u"attributes")
			canbesold,rareCounts,attributes_dict,shouldbuy,sellratio,buyratio=self.analyze_detail(attributes)
			if not canbesold:
				print("触发限制条件。不能卖出")
				return

			maxprice=0
			maxvalue=None
			for i,label in enumerate(['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]):
				value=attributes_dict[label]
				try:					
					# print(label,value,self.marketprice[rareCounts][label][value])
					if self.marketprice[rareCounts][label][value]>maxprice:
						maxprice=self.marketprice[rareCounts][label][value]
						maxvalue=value
				except Exception as e:
					pass

			if attributes_dict[u'体型']==u'天使' and  attributes_dict[u'眼睛']==u'白眉斗眼' and attributes_dict[u'嘴巴']==u'樱桃':
				return
				maxprice*=10
			elif attributes_dict[u'体型']==u'天使' and  attributes_dict[u'眼睛']==u'白眉斗眼':
				return
				maxprice*=6			
			print(maxvalue,maxprice)

	
			sellprice_=round(maxprice*1.6*float(random.uniform(1, 1.3)),0)
			# print(sellprice_)
			data = {
				"petId":petId,
				"requestId": int(time.time() * 1000),
				"amount":sellprice_,
				"appId":1,
				"tpl":"",
				"timeStamp":"",
				"nounce":"",
				"token":""				
			}
			page = requests.post("https://pet-chain.baidu.com/data/market/salePet", data=json.dumps(data),
								 headers=self.headers, verify=False)
			resp = page.json()			
			if resp.get(u"errorMsg") == u"success":
				print('put on shelf:','rareDegree',rareDegree,'amount',sellprice_,"petId",petId,'rareCounts',rareCounts)
			else:
				print("petId",petId,resp.get(u"errorMsg"),resp.get(u"errorNo"))
		except Exception as e:
			print(e) 
			# raise   


def demonstrate_hitory_market(rareDegree=1):
	df_list=[]
	df_sample=pd.DataFrame()
	bins_map={
		0: range(0,500,50),
		1: range(0,1000,100),
		2: range(0,2000,200),
		3: range(0,30000,3000),
		4: range(0,200000,20000),
	}
	for i in range(24,0,-1):
		timeslice=datetime.datetime.now()-datetime.timedelta(hours=i)
		timeslice_str=timeslice.strftime( '%Y-%m-%d_%H' )
		filepath=os.path.join(os.getcwd(),'record','{}.csv'.format(timeslice_str))
		try:
			df=pd.read_csv(filepath,  sep='\t')
			# df_sample=df.loc[0:]
		except Exception as e:
			print('读取上{}个小时记录失败'.format(i))
			# df=df_sample
		else:
			print('读取上{}个小时记录成功'.format(i))
			

			df.drop_duplicates(subset =['id',],inplace=True) 
			df.sort_values(by='amount',ascending=True,inplace=True)
			df['timeslice']=timeslice_str
			# for i in range(5):
			#	 if i in[0,]:
			#		 continue
			df_tmp=df[df.rareDegree==rareDegree]	
			df_list.append(df_tmp)
	df_all=pd.concat(df_list)

	g = sns.FacetGrid(df_all, col="timeslice", col_wrap=6, )#size=None
	g = g.map(plt.hist, "amount", bins=bins_map[rareDegree], color="m")		
	plt.show()		

def analyze_attribute_premium():
	inputfilepath='record_letsgogold/v3_{}.csv'.format((datetime.datetime.now()-datetime.timedelta(hours=1)).strftime( '%Y-%m-%d_%H' ))
	df=pd.read_csv(inputfilepath,  sep='\t')
	df=df[df.rareCounts==2]
	# df.fillna('average')
	# print(df.head(3))

	# g = sns.FacetGrid(df, col="rareCounts", col_wrap=1, )#size=None
	# g = g.map(plt.hist, "amount", bins=range(0,400000,2000), color="m")		
	# plt.show()	

	
	#calculate mean
	for i,label in enumerate(['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]):
		print(label)
		df[label]=df[label].fillna('other')
		values=list(set(list(df[label])))
		for value in values:
			print(value,np.percentile(list(df[df[label]==value]['amount']),2),end=' ')
		print('\n')

	# #calculate mean
	# mask=(df.amount<200000)
	# df=df.loc[mask]
	# # print(df.head(5))
	# for i,label in enumerate(['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]):
	# 	print(label)
	# 	df[label]=df[label].fillna('other')
	# 	values=list(set(list(df[label])))
	# 	for value in values:
	# 		print(value,round(df[df[label]==value]['amount'].mean()/1000,0)*1000,end=' ')
	# 	print('\n')
	
	# plot grid
	for i,label in enumerate(['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]):
		print(label)
		plt.figure(i-1)
		# pltz = PyplotZ()
		g = sns.FacetGrid(df, col=label, col_wrap=5, )#size=None
		g = g.map(plt.hist, "amount", bins=range(0,200000,2000), color="m")		
		# pltz.title("数据图")
		# pltz.legend()
	plt.show()			


def analyze():
	# demonstrate_hitory_market(3)
	analyze_attribute_premium()

def get_market_from_letsgogold():
	headers_='''
		GET http://47.104.126.11/letsgo/data.php?_=1519734597498 HTTP/1.1
		Host: 47.104.126.11
		Connection: keep-alive
		Accept: application/json, text/javascript, */*; q=0.01
		Origin: http://letsgo.gold
		User-Agent: Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36
		Referer: http://letsgo.gold/
		Accept-Encoding: gzip, deflate
		Accept-Language: zh-CN,zh;q=0.9


		HTTP/1.1 200 OK
		Server: nginx/1.12.1
		Date: Tue, 27 Feb 2018 13:04:14 GMT
		Content-Type: text/html; charset=UTF-8
		Connection: keep-alive
		X-Powered-By: PHP/5.6.31
		Access-Control-Allow-Origin: *
		Content-Length: 3076386


	'''
	headers={}
	for line in headers_.split("\n"):
		if line.strip()=='':
			continue
		splited = line.strip().split(":")
		key = splited[0].strip()
		value = ":".join(splited[1:]).strip()
		headers[key] = value	
	# print(headers)
	
	def get_petid(str):
		str=str.split('petId=')[-1]
		str=str.split('&appId=')[0]
		str=str.split('&validCode=')[0]

		return str
	def get_attributes_dict(x,y):
		x=x.split(',')
		y=y.split(',')
		attributes_dict=dict(zip(x,y))
		# print(attributes_dict)
		return attributes_dict
	def get_key_value(dict,label):
		try:
			return dict[label]
		except Exception as e:
			return None

	while True:
		try:
			pass

			data = {
				"_": int(time.time() * 1000),
			}

			page = requests.post("http://47.104.126.11/letsgo/data.php", headers=headers,
								 data=json.dumps(data), verify=False, timeout=2)

			# self.headers['Referer'] = "https://pet-chain.baidu.com/chain/detail?channel=market&petId={}&appId=1&validCode={}".format(pet_id, pet_validCode)
			# page = requests.post("https://pet-chain.baidu.com/data/txn/create", headers=self.headers,
			# 					 data=json.dumps(data), verify=False, timeout=2)
			resp = page.json()
			df=pd.DataFrame(resp.get('data'))
			df.columns=['petid','petUrl','amount','rareDegree','rareCounts','id','typekey','typevalue']

			df=df[df.rareCounts>'1']
			
			df['petid']=list(map(lambda x:get_petid(x),df['petid']))
			#
			df['dict']=list(map(lambda x,y:get_attributes_dict(x,y),df['typekey'],df['typevalue']))
			for label in ['体型','花纹','眼睛','眼睛色','嘴巴','肚皮色','身体色','花纹色',]:
				df[label]=list(map(lambda x:get_key_value(x,label),df['dict']))
			
			df.drop(labels=['petUrl','dict','typekey','typevalue'],axis=1,inplace=True)
			# print(df.head(3))

			filename='record_letsgogold/v3_{}.csv'.format(datetime.datetime.now().strftime( '%Y-%m-%d_%H' ))
			if os.path.exists(filename):
				df.to_csv(filename, sep='\t', encoding='utf-8',index=False,mode='a',header=False)
				# print(filename,'added')
			else:
				df.to_csv(filename, sep='\t', encoding='utf-8',index=False,mode='a',header=True)
				print(filename,'created')
			print(datetime.datetime.now(),'success')
		except Exception as e:
			print(datetime.datetime.now(),'fail')
			print(e)
			# raise e
		time.sleep(60*0.6)

def generate_petid_indexing():
	filedir=os.path.join(os.getcwd() ,'record_letsgogold')


	for file in os.listdir(filedir):
		if file[:3]!='v3_':
			continue
		print(file,end=' ')
		df=pd.read_csv(os.path.join(filedir,file),  sep='\t')
		df=df[df.rareCounts>=5]
		df.petid=list(map(lambda x:str(x).split('petId=')[-1].split('&appId=')[0].split('&validCode=')[0],df.petid))
		df.drop_duplicates(subset =['petid',],inplace=True) 
		print(len(df.index))
		filename=os.path.join(filedir,'sum.csv')
		if os.path.exists(filename):
			df.to_csv(filename, sep='\t', encoding='utf-8',index=False,mode='a',header=False)
			# print(filename,'added')
		else:
			df.to_csv(filename, sep='\t', encoding='utf-8',index=False,mode='a',header=True)
			print(filename,'created')

	# filename=os.path.join(filedir,'sum.csv')
	# df=pd.read_csv(filename,  sep='\t')
	# df.drop(labels=['amount','rareDegree'],axis=1,inplace=True)
	# df.drop_duplicates(subset =['petid',],inplace=True) 
	# df.to_csv(filename, sep='\t', encoding='utf-8',index=False,header=True)
	# return


def trade():
	pass
			
if __name__ == "__main__":
	# get_market_from_letsgogold()
	# os._exit(0)
	# analyze()
	# os._exit(0)
	# generate_petid_indexing()
	# os._exit(0)
	
	# time.sleep(60*3)

	pc = PetChain()
	pc.run()
	# pc.bat_get_veri_code()
	while True:
		t1=time.time()
		pc.modify_inventory()
		t2=time.time()
		print('takes {} seconds'.format(t2-t1))
		print(datetime.datetime.now())
		time.sleep(60*3)
