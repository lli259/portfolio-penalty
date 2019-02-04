#result.txt to result.xls
import xlwt
alllines=[]
with open('result.txt',"r") as f:
	alllines=f.readlines()
print(len(alllines))
alllines=[" "]+[i[:-1] for i in alllines]
print(len(alllines))
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('Worksheet')
def wr(i,j,m):
	worksheet.write(i-1, ord(j)-97, label=m)

def wrl(a,b,c,d):
	for i in range(b-a+1):
		wr(c+i, d,  alllines[a+i])

def wrl2(a,c,d):
	for i in a:
		wr(c, d,  i)
		c+=1
wrl(15,19,1,"a")

##
addlineDset={}
addlineDset[0]=0
addlineDset[120-21]=26
addlineDset[219-21]=52
for addsetline in [0, 120-21,219-21]:
	wrl(21+addsetline,21+addsetline,7+addlineDset[addsetline],"a")

	a=[i.split(" ")[0] for i in alllines[22+addsetline:28+addsetline]]
	wrl2(a,8+addlineDset[addsetline],"a")

	a=["".join(i.split(" ")[1:]) for i in alllines[22+addsetline:28+addsetline]]
	wrl2(a,8+addlineDset[addsetline],"b")

	a=alllines[28+addsetline]
	wrl2([a],14+addlineDset[addsetline],"a")

	a=alllines[29+addsetline]
	wrl2([a],14+addlineDset[addsetline],"b")


	row={}
	row[0]="f"
	row[0+addsetline]=row[0]
	row[1]="g"
	row[1+addsetline]=row[1]
	row[60-31]="i"
	row[60-31+addsetline]=row[60-31]
	row[60-30]="j"
	row[60-30+addsetline]=row[60-30]
	row[89-31]="l"
	row[89-31+addsetline]=row[89-31]
	row[89-30]="m"
	row[89-30+addsetline]=row[89-30]

	for addline in [0+addsetline,60-31+addsetline,89-31+addsetline]:
		print(31+addline)
		print(32+addline)
		print(6+addlineDset[addsetline])
		print(row[addline])
		wrl(31+addline,32+addline,6+addlineDset[addsetline],row[addline])
		a=[alllines[34+addline+2*i] for i in range(6)]
		wrl2(a,8+addlineDset[addsetline],row[addline])

		if addline==0+addsetline:
			a=[i.split(" ")[0] for i in alllines[47+addline:51+addline]]
			wrl2(a,16+addlineDset[addsetline],"a")

		a=["".join(i.split(" ")[1:]) for i in alllines[47+addline:51+addline]]
		wrl2(a,16+addlineDset[addsetline],row[addline])

		if addline==0+addsetline:
			a=alllines[53+addline].split(" ")[0]
			wrl2([a],21+addlineDset[addsetline],"a")

		a=alllines[53+addline].split(" ")[1]
		wrl2([a],21+addlineDset[addsetline],row[addline])

		if addline==0+addsetline:
			a=alllines[54+addline].split(" ")[0]
			wrl2([a],22+addlineDset[addsetline],"a")

			a=alllines[54+addline].split("{")[1].split("}")[0].split(",")
			wrl2(a,23+addlineDset[addsetline],"a")

		a=alllines[55+addline].split(" ")[0]
		wrl2([a],22+addlineDset[addsetline],row[addline])


		a=alllines[55+addline].split("{")[1].split("}")[0].split(",")
		wrl2(a,23+addlineDset[addsetline],row[addline])


		a=alllines[56+addline].split(" ")[0]
		wrl2([a],22+addlineDset[addsetline],row[addline+1])

		a=alllines[56+addline].split("{")[1].split("}")[0].split(",")
		wrl2(a,23+addlineDset[addsetline],row[addline+1])

		if addline==0+addsetline:
			a=[i.split(" ")[0] for i in alllines[57+addline:59+addline]]
			wrl2(a,29+addlineDset[addsetline],"a")
		a=[i.split(" ")[1] for i in alllines[57+addline:59+addline]]
		wrl2(a,29+addlineDset[addsetline],row[addline])






workbook.save('result.xls')
