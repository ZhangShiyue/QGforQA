# download QQP data
import os
import urllib.request
os.system("mkdir LIB/qqp")
urllib.request.urlretrieve(
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP-clean.zip?alt=media&token=11a647cb-ecd3-49c9-9d31-79f8ca8fe277",
    "LIB/qqp/QQP.zip")
os.system("cd LIB/qqp; unzip QQP.zip")