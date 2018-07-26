import glob
import sys
import csv

filePath = "C:/Users/Sava/Documents/Movie-sentiment-review/Unprocessed data/aclImdb/train/pos/*" #sys.argv[1]
label = "1" #sys.argv[2]
outFileName = "C:/Users/Sava/Documents/Movie-sentiment-review/Data Parsing/labeledData.csv" #sys.argv[3]

outFile = csv.writer(open(outFileName, "w", newline=''))
files = glob.glob(filePath)   
for name in files:
    try:
        with open(name) as inFile: 
            outFile.writerow([inFile.read(),label])
    except:
        print(name)


filePath = "C:/Users/Sava/Documents/Movie-sentiment-review/Unprocessed data/aclImdb/train/neg/*" #sys.argv[1]
label = "0" #sys.argv[2]

files = glob.glob(filePath)   
for name in files:
    try:
        with open(name) as inFile: 
            outFile.writerow([inFile.read(),label])
    except:
        print(name)

# filePath = "C:/Users/Sava/Documents/Movie-sentiment-review/Unprocessed data/aclImdb/test/pos/*" #sys.argv[1]
# label = "1" #sys.argv[2]

# files = glob.glob(filePath)   
# for name in files:
#     try:
#         with open(name) as inFile: 
#             outFile.writerow([inFile.read(),label])
#     except:
#         print(name)

# filePath = "C:/Users/Sava/Documents/Movie-sentiment-review/Unprocessed data/aclImdb/test/neg/*" #sys.argv[1]
# label = "0" #sys.argv[2]

# files = glob.glob(filePath)   
# for name in files:
#     try:
#         with open(name) as inFile: 
#             outFile.writerow([inFile.read(),label])
#     except:
#         print(name)