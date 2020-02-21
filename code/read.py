import csv

def read(filepath):
    x=list()
    y=list()
    names=list()
    count=0
    with open (filepath) as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        for row in csv_reader:
            if count==0:
                names.append(row)
                count+=1
            else:
                temp=list()
                for i in range (len(row)):
                    if i == len(row)-1 or i== len(row)-2:
                        if i ==len(row) -1:
                            y.append(row[i])
                    else:
                        temp.append(row[i])
                x.append(temp)
                count+=1
    return names,x,y,count
