from pathlib import Path

trainPath = '../textFile/train.txt'
testPath = '../textFile/test.txt'
DBPath = '../dataset/DB/jpeg/'
QueryPath = '../dataset/Query/jpeg/'

f = open(trainPath,'w')
p = Path(DBPath)
p = sorted(p.glob("*.jpg"))
count = 0
for filename in p:
    f.write(str(filename) + ' ' + str(count/10) + '\n')
    count += 1
f.close()

f = open(testPath,'w')
p = Path(QueryPath)
p = sorted(p.glob("*.jpg"))
count = 0
for filename in p:
    if filename.name[0:1] == 'r':
        num = -1
    else:
        num = int(filename.name[0:2])
    f.write(str(filename) + ' ' + str(num) + '\n')
    count += 1
f.close()