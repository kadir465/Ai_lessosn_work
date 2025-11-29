import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class RestornatRecommed:
    def __init__(self,data_path):
        self.data=pd.read_csv(data_path)
        self.preprocess()
        self.creat_ml_model()

    def preprocess(self):

        self.data["mutfak"]=self.data["mutfak"].astype(str).str.strip().str.lower()
        self.data["Restoran Adı"]=self.data["Restoran Adı"].astype(str).str.strip()
        self.data["Restoran_Adi_Lower"]=self.data["Restoran Adı"].astype(str).str.lower()

        self.data["degerlendirme"]=pd.to_numeric(self.data["degerlendirme"],errors="coerce").fillna(0)
        self.data["fiyat aralgı kod"]=pd.to_numeric(self.data["fiyat aralıgı kod"],errors="coerce").fillna(2)

        scaler=MinMaxScaler()
        self.data["norm_puan"]=scaler.fit_transform(self.data[["Puan"]])
        self.data["norm_degerlendirme"]=scaler.fit_transform(self.data[["degerlendirme"]])
        norm_fiyat=scaler.fit_transform(self.data[["fiyat aralıgı kod"]])
        self.data["norm_fiyat_score"]=1-norm_fiyat
        self.data["combine_feature"]=self.data["mutfak"]+" "+self.data["Restoran_Adi_Lower"]

    def creat_ml_model(self):
        self.tfidf=TfidfVectorizer(stop_words=None)
        self.tfidf_matrix=self.tfidf.fit_transform(self.data["combine_feature"])
    
    def recommed_resturant(self,user_query,top=5):
        user_query=user_query.lower().strip()
        query_vect=self.tfidf.transform([user_query])
        similarity_scor=cosine_similarity(query_vect,self.tfidf_matrix)

        self.data["similarity_scor"]=similarity_scor[0]

        filtered_result=self.data[self.data["similarity_scor"]>0].copy()

        if filtered_result.empty:
            return None
        
        w_puan=0.5
        w_degerlendirme=0.3
        w_fiyat=0.2

        filtered_result["wighted_scor"]=(
            (filtered_result["norm_puan"]*w_puan)+
            (filtered_result["norm_degerlendirme"]*w_degerlendirme)+
            (filtered_result["norm_fiyat_score"]*w_fiyat)
        )

        last_result=filtered_result.sort_values(by="wighted_scor",ascending=False).head(top)

        return last_result

dosya_yolu = r"C:\Users\kadir\OneDrive\Masaüstü\Software_project\restorant_advice_system\restorant_bilgileri.csv"
recommed=RestornatRecommed(dosya_yolu)

print("Öneri Sistemimiz sadece Döner , Kebap ve Pizza restorntlarını kapsıyor\n")
choice=input("bugün ne yemek istiyorsun :) ")
print(choice)
sonuc=recommed.recommed_resturant(choice)

print(f"---'{choice}' içinml ve ağırlık scor önerileri")
if sonuc is not None:
    cols=['Restoran Adı', 'Puan', 'degerlendirme', 'fiyat aralıgı kod', 'wighted_scor']
    print(sonuc[cols])
    suggest_restorant=sonuc['Restoran Adı'].tolist()
    print("\nLLM Modeline Gidecek Liste:", suggest_restorant)
else:
    print("aradığınız kriterde restornt bulunamdaı ")

