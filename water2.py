from PIL import Image,ImageFile,ImageFilter,ImageEnhance
import os
from itertools import groupby
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
# from PIL import 
import pytesseract
import time
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.neighbors.kde import KernelDensity
import shutil
import seaborn as sns

# from scipy.stats import gaussian_kde #as kde
# from sklearn.neighbors import KernelDensity

ImageFile.LOAD_TRUNCATED_IMAGES = True

show_detail=False
# show_detail=True

global clf,kde_comb,kde_letter
clf=None
kde_comb=None
kde_letter=None


def reg_line(img):
    a = img.load()
    M,N = img.size

    b=np.ones(M)
    # pr(b)
    # print(len(b))
    # for i in range(M):
    #    b[i]=-1
    # pr(b)

    threshold = 100;  
    offset = 11;  

    Black=0
    White=255

    linelist=[]
    def genLine(n):
        print(n)
        if (n < offset) :      
            b[n] = -1;  
            genLine(n+1);  
        
        if (n == M)  :    
            print(b) 
            linelist.append(b)
        
        if (n==offset)  :     
            for j in range(N) :    
                if a[offset,j] == Black:
              
                    b[offset] = j;  
                    genLine(n+1);  
                
            
        
        if (n > 0 and n < M)  :
      
            hasMore = 0;  
            if (a[n,b[n-1]] == Black):
                b[n] = b[n-1];  
                hasMore = 1;  
                genLine(n+1);  
            
            else  :
          
                if (b[n-1] > 0 and a[n,b[n-1]-1] == Black):
                    b[n] = b[n-1]-1;  
                    hasMore = 1;  
                    genLine(n+1);  
                
                if (b[n-1] < N-1 and a[n,b[n-1]+1] == Black):
                    b[n] = b[n-1]+1;  
                    hasMore = 1;  
                    genLine(n+1);  
                
            
            if (n - offset > threshold and hasMore == 0)  :    
                print(b) 
                linelist.append(b)
        
    genLine(0);
    print(len(linlist))


def reg_main_line(img):
    img=img.filter(ImageFilter.MedianFilter)  

    if show_detail==True:
        img.save('image/' +  "2.0-deleteline.png")

    pixdata = img.load()
    w, h = img.size

    # print(type(pixdata))

    maxLineWidth=4
    Black=0
    White=255

    top_len=[0]
    top_len_x=[-1]
    top_len_y=[-1]
    matrix = [[None]*h for i in range(w)] 
    # print(len(matrix))
    # print(matrix[134][59])
    
    # print(pixdata[160,60])
    


    for x in range(w):
        for y in range(h):
            if pixdata[x,y]==White:
                continue
            node={
                # 'x':
                # 'y':
                'accu_delta_y':0,
                'len':1,
                'previous_node_x':None,
                'previous_node_y':None,
                'is_start':0,
            }
            
            if x==0:
                node['is_start']=1
            else:
                max_len=0
                accu_delta_y=0
                result_bias=None
                for y_bias in [-1,0,1]:
                    if (y+y_bias <0) or (y+y_bias >=h):
                        continue
                    if pixdata[x-1,y+y_bias] == White:
                        continue
                    if matrix[x-1][y+y_bias]['len']>max_len:
                        result_bias=y_bias
                        max_len=matrix[x-1][y+y_bias]['len']+1
                        accu_delta_y=matrix[x-1][y+y_bias]['accu_delta_y']+abs(y_bias)


                    elif (matrix[x-1][y+y_bias]['len']==max_len) and (  matrix[x-1][y+y_bias]['accu_delta_y']+abs(y_bias)> accu_delta_y   ):
                        result_bias=y_bias
                        max_len=matrix[x-1][y+y_bias]['len']+1
                        accu_delta_y=matrix[x-1][y+y_bias]['accu_delta_y']+abs(y_bias)
                if result_bias is None:
                    node['is_start']=1
                else:
                    node['previous_node_x']=x-1
                    node['previous_node_y']=y+result_bias   
                    node['accu_delta_y']=   accu_delta_y                
                    node['len']=   max_len        
            matrix[x][y]=node
            # if x==127:
            #     print(x,y,matrix[x][y])      
            if max(top_len)<matrix[x][y]['len']:
                top_len.append(matrix[x][y]['len'])
                top_len_x.append(x)
                top_len_y.append(y)

    # print(top_len[-1],top_len_x[-1],top_len_y[-1])
    
    x=top_len_x[-1]
    y=top_len_y[-1]
    while True:
        # print(matrix[x][y])
        this_bottom=y
        try:
            while pixdata[x,this_bottom-1]==Black:
                this_bottom=this_bottom-1
        except Exception as e:
            pass
        this_top=y
        try:
            while pixdata[x,this_top+1]==Black:
                this_top+=1
        except Exception as e:
            pass                
        if this_top-this_bottom+1<=maxLineWidth:
            for y_ in range(this_bottom,this_top+1):
                pixdata[x,y_]=White
        else:
            pass
            # pixdata[x,y]=White
        
        x,y=matrix[x][y]['previous_node_x'],matrix[x][y]['previous_node_y']

        # print(this_top-this_bottom+1,x,y,)
        if x is None:
            break



    # img.show()
    return img
        

def binarizing(img, threshold):
    
    # img.save('water/' + fileName + "-original.png")
    # 
    img = img.convert("L")  # ת?Ҷ?
    # img.save('water/' + fileName + "-grey.png")
    pixdata = img.load()
    w, h = img.size
    # print(w,h)
    
    if show_detail==True:
        img.save('image/' +  "0-original.png")

    for y in range(h):
        if pixdata[0, y]==0:
            # crop image
            # h=y
            # region=(0,0,w,h)
            # img = img.crop(region)
            # pixdata = img.load()
            
            # modify image
            for y_ in range(y,h):
                for x_ in range(w):
                    pixdata[x_,y_]=255
            break
        # print(y,pixdata[0, y])
    # def f_or(x,y):
    #    return x or y
    # print([pixdata[x, 0]] for x in range(w))
    #    out=reduce(f_or, [pixdata[x, y]] for x in range(w))
    #    print(y,out)





    # for x in range(h):
    #   print(pixdata[x, 0] )
    # print(pixdata[0, :])
    
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] >= threshold:
                pixdata[x, y] = 255
            elif pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    
    if show_detail==True:
        img.save('image/' +  "1-binary.png")
    
    img=reg_main_line(img)
    if show_detail==True:
        img.save('image/' +  "2-deleteline.png")
   



    pixdata = img.load()
    for y in range(h):
        for x in range(35):
            pixdata[x,y]=255
        # for x in range(w-w):
        #     pixdata[x,y]=255
    
    # img.show()
    return img,w, h



def delete_line(img2,limit):
    img=img2.copy()
    pixdata = img.load()
    w, h = img.size 
    for x in range(1,w-1):
        for y in range(1,h-1): 
            if pixdata[x,y]==255:
                continue
            try:
                count=0
                for i,j in itertools.product([-1,0,1],[-1,0,1]):
                    # print(i,j)
                    if pixdata[x+i,y+j]>0:
                        count+=1
                if count>limit:
                    pixdata[x,y]=255
            except Exception as e:
                raise e  

    img.save('water/' + fileName + "-deleteline{}.png".format(limit))
    return img

def vertical(img):
    
    pixdata = img.load()
    w, h = img.size
    result = []
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x, y] == 0:
                black += 1
        result.append(black)
    
    if show_detail==True:
        plt.bar(range(len(result)),result)
        # plt.show()
        plt.savefig('image/' +  "4-vertical.png")
        plt.close()


    # bandwidth = 1.5
    # density2 = gaussian_kde(result, bw_method=bandwidth)
    # xs = range(len(result))
    # plt.plot(xs,density2.evaluate(xs), label='bw_method={}'.format(bandwidth))
    # plt.savefig('image/' +  "4.1-kde.png")
    # plt.close()

    return result


def get_cuts2(result,img,fileName=None):
    global kde_comb,kde_letter
    if kde_comb is None:
        kde_comb = joblib.load('data/is_legal_comb_kde.pkl')      
    if kde_letter is None:
        kde_letter = joblib.load('data/is_one_letter_kde.pkl')     

    result_df=pd.DataFrame(result,columns=['bar'])
    image_cut_list=[]
    w,h=img.size

    Dmin=8
    Dmax=28
    prob_letter_threshold=-9
    
    l, r = 0, 0
    flag = False
    cuts = []
    for i, count in enumerate(result):
        
        if flag is False and count > 0:
            l = i
            flag = True
        if flag and count == 0:
            r = i - 1
            flag = False
            
            cuts.append([l,r])
    cuts_np=np.array(cuts).reshape(-1, 2)
    # print(cuts_np,type(cuts_np),cuts_np.shape)
    cuts_df=pd.DataFrame(cuts_np,columns=['left','right'])
    cuts_df['height']=list(map(lambda x,y:result_df.loc[x:y,'bar'].max(),cuts_df.left,cuts_df.right))
    cuts_df['minindex']=list(map(lambda x,y:result_df.loc[x:y,'bar'].idxmin(),cuts_df.left,cuts_df.right))
    cuts_df['length']=list(map(lambda x,y:y-x+1,cuts_df.left,cuts_df.right))
    cuts_df['prob_letter']=kde_letter.score_samples(cuts_df[['left','right']].as_matrix())

    # 'height','minindex','length'
    # print(cuts_df)

    mask=(cuts_df.height>10)&(cuts_df.prob_letter>prob_letter_threshold)
    df_tmp=cuts_df.loc[mask]
    if len(df_tmp.index)==4:
        # print("absolute 4 cuts, no need to process anymore")
        df_tmp.reset_index(drop=True, inplace=True)
        cuts_df=df_tmp
    else: 
        # print(len(df_tmp.index),"split or combine")
        
        i=0
        looplen=len(cuts_df.index)
        while True:
            if i>=looplen:
                break
            if i not in cuts_df.index:
                continue
            # print(i)
            # print(cuts_df.loc[i,:])
            if cuts_df.loc[i,'length']>Dmax and cuts_df.loc[i,'prob_letter']<prob_letter_threshold:
                # print('split',i)
                left,right,localmin=cuts_df.loc[i,['left','right','minindex']]
                
                localmin=int(result_df.loc[left+6:right-6,'bar'].idxmin())
             
                score1=cuts_df.loc[i,'prob_letter']
                score2=kde_letter.score_samples([[left,localmin]]).max()
                score3=kde_letter.score_samples([[localmin+1,right]]).max()
                # print(score1,score2,score3)
                if score1<score2 and score1<score3:
                    pass
                    cuts_df=cuts_df.append(pd.DataFrame(np.array([left,localmin]).reshape(-1, 2),columns=['left','right'],index=[i+100]))
                    cuts_df=cuts_df.append(pd.DataFrame(np.array([localmin+1,right]).reshape(-1, 2),columns=['left','right'],index=[i+200]))
                    cuts_df.drop(labels=i,axis=0,inplace=True)
                    i+=1
                    continue
            if (i<len(cuts_df.index)-1) and i>0:
                # if cuts_df.loc[i,'height']>10   and cuts_df.loc[i,'length']<Dmin and cuts_df.loc[i+1,'length']<Dmin    and cuts_df.loc[i+1,'height']>10 and (cuts_df.loc[i+1,'left']-cuts_df.loc[i,'right']-1<=7) and (cuts_df.loc[i+1,'length']+cuts_df.loc[i,'length']<Dmax):
                score1=cuts_df.loc[i,'prob_letter']
                score2=-100
                score3=-100
                if (i-1)  in cuts_df.index:
                    score2=kde_letter.score_samples([[cuts_df.loc[i-1,'left'],cuts_df.loc[i,'right']]]).max()
                if (i+1)  in cuts_df.index:
                    score3=kde_letter.score_samples([[cuts_df.loc[i,'left'],cuts_df.loc[i+1,'right']]]).max()
                # print(score1,score2,score3)
                if ((i-1)  in cuts_df.index) and score2>score3 and score2>score1 and score2>prob_letter_threshold:
                    print('combine',i-1,i)
                    # print(pd.DataFrame(np.array([cuts_df.loc[i-1,'left'],cuts_df.loc[i,'right']]).reshape(-1, 2),columns=['left','right']))
                    df_append=pd.DataFrame(np.array([cuts_df.loc[i-1,'left'],cuts_df.loc[i,'right']]).reshape(-1, 2),columns=['left','right'],index=[i+100])
                    # print(df_append)
                    cuts_df=cuts_df.append(df_append)
                    cuts_df.drop(labels=[i-1,i],axis=0,inplace=True)
                    # print(cuts_df)
                    i+=1
                    continue
                if ((i+1)  in cuts_df.index) and score3>score2 and score3>score1 and score3>prob_letter_threshold:
                    print('combine',i,i+1)
                    cuts_df=cuts_df.append(pd.DataFrame(np.array([cuts_df.loc[i,'left'],cuts_df.loc[i+1,'right']]).reshape(-1, 2),columns=['left','right'],index=[i+100]))             
                    cuts_df.drop(labels=[i,i+1],axis=0,inplace=True)
                    # print(cuts_df)
                    i+=2                    
                    continue
            i+=1
            continue
        cuts_df['height']=list(map(lambda x,y:result_df.loc[x:y,'bar'].max(),cuts_df.left,cuts_df.right))
        cuts_df['length']=list(map(lambda x,y:y-x+1,cuts_df.left,cuts_df.right))
        # print(cuts_df)
        cuts_df.sort_values(by=['height'],ascending=[0], inplace=True)

        cuts_df=cuts_df[:4]
        cuts_df.sort_values(by=['left'],ascending=[1], inplace=True)
        cuts_df.reset_index(drop=True, inplace=True)        

    # print(cuts_df)

    for i  in range(len(cuts_df.index)):
        left,right=cuts_df.loc[i,['left','right']]
        image_cut = Image.new('L', (int(right-left+1), h), (255))
        image_cut = img.crop((left,0,right,h))        
        image_cut_list.append(image_cut)
        if show_detail==True:
            image_cut.save('image/' +  "5-cut{}.png".format(i))
    # for i,cut in enumerate(cuts):
    
    #         localmin=result_df.loc[left:right+1,'bar'].idxmin()
    #         cuts.remove(cut)
    #         cuts.append([left,localmin])
    #         cuts.append([localmin+1,right])   
    # cuts=sorted(cuts,key=lambda x:x[0])
    
    # for i in range(len(cuts)-1):
    #     maxheight_left=result_df.loc[cuts[i][0]:(cuts[i][1]+1),'bar']
    #     maxheight_right=result_df.loc[(cuts[i+1][0]):(cuts[i+1][1]+1),'bar']
    #     if maxheight_left<15:
    #         continue
    #     if maxheight_right<15:
    #         continue
    #     if cuts[i+1][0]-(cuts[i][1]+1)>2:
    #         continue




    # for i,cut in enumerate(cuts):    
    
    #         left_distance=cut[]
    #         if cut:
    #             pass
    #     else:
    #         cuts_adjuste.append([left,right])

    
    # for i,cut in enumerate(cuts):
    #     pass

    outputstring=''

    if fileName is not None:
        global clf
        if clf is None:
            pass
            clf = joblib.load('data/letter.pkl')  
        data=[]
        for image_cut in image_cut_list:
            im = np.array(image_cut)
            cv_gray=im[:, :].copy()             
            data.append(getletter(cv_gray))
        data=np.array(data)
        oneLetter = clf.predict(data)
        if "".join(oneLetter) != fileName.split('.')[0]:
            
            return outputstring,image_cut_list                      
        
        y=fileName.split('.')[0]
        for i,image_cut in enumerate(image_cut_list):
            recdString =   "{}{}.png".format(y,str(round(time.time(),0)))
            w,h=image_cut.size
            # if (w>10 and y[i] not in ['t','j','i',]) or (w>8 and y[i] in ['t','j','i',]):
            path = os.path.join(os.getcwd(),'temp' ,y[i] )
            # else:
            #     path = os.path.join(os.getcwd(),'other'  )
            if not os.path.exists(path):
                os.mkdir(path)
            imgPath = os.path.join(path,recdString)
            image_cut.save(imgPath)
        
        df_location=pd.DataFrame()
        for i  in range(len(cuts_df.index)):
            left,right=cuts_df.loc[i,['left','right']]
            # print(left,right)
            df_location['left{}'.format(i)]=[left]
            df_location['right{}'.format(i)]=[right]
        # print(df_location)
        path=r'data/location.csv'
        if not os.path.exists(path):
            df_location.to_csv(path, sep='\t',mode='a', encoding='utf-8',index=False,header=True)
        else:
            df_location.to_csv(path, sep='\t',mode='a', encoding='utf-8',index=False,header=False)

    return outputstring,image_cut_list


def get_cuts(result,img,fileName=None):
    w,h=img.size
    result=pd.DataFrame(result,columns=['bar'])
    nodes=[ 43,66,87,110,130,]
    threshold=2
    window=6
    # derivative=10
    cuts=[]
    
    localmin_list=[]
    image_cut_list=[]
    for node in nodes:
        # print(node,node-window,node+window)
        df=result.loc[node-window:node+window,:]
        # print(df)
        # value = alli.min()
        # index = alli.argmin()
        localmin=df.bar.min()
        left=df[df.bar<=localmin+threshold].index.min()-1
        right=df[df.bar<=localmin+threshold].index.max()+1
        localmin_list.append((left+right)/2)
        # print(left,right)
        cuts.append([left,right])
    with open('localmin.txt', 'a+') as f:
        for localmin in localmin_list:
            # print(localmin)
            f.write('{}{}'.format(localmin,'\t'))   
        f.write('\n') 
    for num in range(len(nodes)-1):
        left=cuts[num][1]
        right=cuts[num+1][0]
        # print(left,right)
        image_cut = Image.new('L', (right-left+1, h), (255))
        image_cut = img.crop((left,0,right,h))
        # image_cut.show()
        # 
        if show_detail==True:
            image_cut.save('image/' +  "5-cut{}.png".format(num))
        image_cut_list.append(image_cut)
    
    outputstring=''
    if fileName is not None:
        pass
        y=fileName.split('.')[0]
        for i,image_cut in enumerate(image_cut_list):
            recdString =   "{}{}.png".format(y,str(round(time.time(),0)))
            w,h=image_cut.size
            if (w>10 and y[i] not in ['t','j','i',]) or (w>8 and y[i] in ['t','j','i',]):
                path = os.path.join(os.getcwd(),'temp' ,y[i] )
            else:
                path = os.path.join(os.getcwd(),'other'  )
            if not os.path.exists(path):
                os.mkdir(path)
            imgPath = os.path.join(path,recdString)
            image_cut.save(imgPath)

        #use pytesseract to get the first bunch of samples
        # for image_cut in image_cut_list:
        #    recNum = pytesseract.image_to_string(image_cut, config='-psm 7 outputbase letters')
        #    # print(recNum)
        #    outputstring+=recNum


    # if not os.path.exists('localmin.txt'):
    #    os.makedirs('localmin.txt')

    return outputstring,image_cut_list


def extractLetters(path):
    x = []
    y = []
    
    
    for root, sub_dirs, files in os.walk(path):
        for dirs in sub_dirs:
            
            for fileName in os.listdir(path + '/' + dirs):
                # print(fileName)
                
                fn=path + '/' + dirs + '/' + fileName
                fnimg = cv2.imread(fn,cv2.IMREAD_GRAYSCALE) 
                x.append(getletter(fnimg))
                y.append(dirs)

    return x, y



# 提取SVM用的特征值, 提取字母特征值
def getletter(fnimg):
    w,h=18,60
     # 读取图像
    img = cv2.resize(fnimg, (w,h))  # 将图像大小调整为18,60
    # cv2.imshow('image',img)
    # cv2.waitKey(0) 
    
    alltz = []
    for now_h in range(0, h):
        xtz = []
        for now_w in range(0, w):
            grey = img[now_h, now_w]
            if grey > 0:
                nowtz = 1
            else:
                nowtz = 0
            xtz.append(nowtz)
        alltz += xtz
    return alltz

# 进行向量机的训练SVM
def trainSVM():
    array = extractLetters('temp')
    # 使用向量机SVM进行机器学习
    letterSVM = SVC(kernel="linear", C=1).fit(array[0], array[1])
    score=letterSVM.score(array[0], array[1])
    print(score)
    # 生成训练结果
    joblib.dump(letterSVM, 'data/letter.pkl')

# 传入测试图片，进行识别测试
def ocrImg(fileName=None):
    clf = joblib.load('data/letter.pkl')
    # p = Image.open('test_img/%s' % fileName)
    data=getletter(r'D:\python\baidu_pet_chain\verifyCode-master\verifyCode\temp\p\aapf1518777391.0.png')
    data=np.array(data).reshape(1, -1)
    oneLetter = clf.predict(data)
    # oneLetter = clf.predict_proba(data)

    print(oneLetter)


def get_split_seq(projection_x):
    split_seq = []
    start_x = 0
    length = 0
    for pos_x, val in enumerate(projection_x):
        if val == 0 and length == 0:
            continue
        elif val == 0 and length != 0:
            split_seq.append([start_x, length])
            length = 0
        elif val == 1:
            if length == 0:
                start_x = pos_x
            length += 1
        else:
            raise Exception('generating split sequence occurs error')
    
    if length != 0:
        split_seq.append([start_x, length])
    return split_seq


def do_split(source_image, starts, filter_ends):
    """
    
    
    
    """
    left = starts[0][0]
    top = starts[0][1]
    right = filter_ends[0][0]
    bottom = filter_ends[0][1]
    pixdata = source_image.load()
    for i in range(len(starts)):
        left = min(starts[i][0], left)
        top = min(starts[i][1], top)
        right = max(filter_ends[i][0], right)
        bottom = max(filter_ends[i][1], bottom)
    width = right - left + 1
    height = bottom - top + 1
    image = Image.new('RGB', (width, height), (255, 255, 255))
    for i in range(height):
        start = starts[i]
        end = filter_ends[i]
        for x in range(start[0], end[0] + 1):
            if pixdata[x, start[1]] == 0:
                image.putpixel((x - left, start[1] - top), (0, 0, 0))
    return image


def drop_fall(img, fileName=None):
    
    
    
    img ,width, height = binarizing(img, 230)

    img=img.filter(ImageFilter.MaxFilter(3))  
    # img.show()
    if show_detail==True:
        img.save('image/' +  "3-MaxFilter.png")



    hist_width = vertical(img)
    


    img=img.filter(ImageFilter.MedianFilter)    
    # img.show()
    if show_detail==True:
        img.save('image/' +  "4-MedianFilter.png")
    

    # reg_line(b_img)


    

    outputstring,image_cut_list=get_cuts2(hist_width,img,fileName)
    # print(outputstring,end=' ')
    return outputstring,image_cut_list
  

def generate_sample():

    for path in ['other','temp']:
        try:
            shutil.rmtree(path) 
        except Exception as e:
            pass 
        if not os.path.exists(path):
            os.makedirs(path)
        # input('{}deleted.go on :'.format(path))
    for root, sub_dirs, files in os.walk(os.path.join(os.getcwd(),'data','captcha_dataset')):
        for fileName in files:
            try:
                if fileName.split('.')[-1]!='jpg':
                    continue
                print(fileName.split('.')[0],end=' ')
                t1=time.time()
                p = Image.open(os.path.join(os.getcwd(),'data','captcha_dataset',fileName) )
                outputstring,image_cut_list=drop_fall(p, fileName)
                t2=time.time()
                print((outputstring==fileName.split('.')[0]),end=' ')
                print(round(t2-t1,2))
            except Exception as e:
                print( e)


def reg_veri_code(img,clf=None):

    if clf is None:
        pass
        clf = joblib.load('data/letter.pkl')
    # t1=time.time()
    
    outputstring,image_cut_list=drop_fall(img)
    # b_img ,width, height = binarizing(img, 230) 
    
    # hist_width = vertical(b_img)

    # outputstring,image_cut_list=get_cuts(hist_width,b_img)
    data=[]
    for image_cut in image_cut_list:
        im = np.array(image_cut)
        # cv_img = im.astype(np.uint8)
        # cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        cv_gray=im[:, :].copy() 
        
        data.append(getletter(cv_gray))
    data=np.array(data)
    
    # data=np.array(data).reshape(1, -1)
    oneLetter = clf.predict(data)
    # oneLetter = clf.predict_proba(data)

    print("".join(oneLetter))   
    # t2=time.time()
    # print(t2-t1)
    return "".join(oneLetter)

def analyze_split():
    path=r'data/location.csv'
    df=pd.read_csv(path,  sep='\t')
    
    # #draw seperately
    # for i in range(4):
    #     g = sns.jointplot("left{}".format(i), "right{}".format(i), data=df,
    #                            kind="kde", space=0, color="g")    
    #     # plt.show()
    #     plt.savefig('data/distribution{}.png'.format(i), dpi=75)

    print('calculate the kde of combination')
    X=df.as_matrix()
    # print(X[:30])
    kde = KernelDensity(kernel='gaussian').fit(X)
    score=kde.score_samples(X)  
    print(np.percentile(score,2))
    joblib.dump(kde, 'data/is_legal_comb_kde.pkl')
    


    df_all_list=[]
    for i in range(4):
        df_all=df.loc[:,["left{}".format(i), "right{}".format(i)]]
        df_all.columns = ['left', 'right']
        df_all_list.append(df_all)
    df_all=pd.concat(df_all_list)
    # #draw together
    # g = sns.jointplot('left', 'right', data=df_all,
    #                        kind="kde", space=0, color="g")    
    # # plt.show()
    # plt.savefig('data/distribution{}.png'.format('all'), dpi=75)    

    print('calculate the kde of one letter')
    X=df_all.as_matrix()
    # print(X[:30])
    kde = KernelDensity(kernel='gaussian').fit(X)
    score=kde.score_samples(X)  
    print(np.percentile(score,2))#-9
    joblib.dump(kde, 'data/is_one_letter_kde.pkl')
    # print(score) 
    # print(score.mean()) 
    # df_all['score']=score
    # df_all.to_csv('data/location_kde.csv', sep='\t',mode='a', encoding='utf-8',index=False,header=True)

if __name__ == '__main__':
    # generate_sample()
    # trainSVM()
    ## ocrImg()
    
    # p = Image.open(r'D:\python\baidu_pet_chain\baidu_lcg-master\data\captcha_wrongmark\pyde.jpg')
    # reg_veri_code(p)
    # 
    # img= Image.open(r'D:\python\baidu_pet_chain\verifyCode-master\verifyCode\water\2jav.jpg')
    # binarizing(img, 200)
    
    # while True:
    #     file=input('bad sample:')
    #     p = Image.open(r'D:\python\baidu_pet_chain\baidu_lcg-master\data\captcha_wrongmark\{}.jpg'.format(file))
    #     reg_veri_code(p)    
        
    # analyze_split()

    path=r'D:\python\baidu_pet_chain\baidu_lcg-master\data\captcha_wrongmark'
    filelist=os.listdir(path)
    filelist=sorted(filelist,key=lambda x:os.stat(path + "/" + x).st_ctime,reverse=True)
    for file in filelist:
        print(file)
        filepath=os.path.join(path,file)
        p = Image.open(filepath)
        reg_veri_code(p) 
        input('go on:')
