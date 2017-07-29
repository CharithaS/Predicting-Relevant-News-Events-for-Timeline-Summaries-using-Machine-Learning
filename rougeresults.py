import subprocess

sumR1=0
sumR2=0
for i in range(1,8):
	output = subprocess.check_output("java -cp C_Rouge/C_ROUGE.jar executiverouge.C_ROUGE ~/SysSummary/DateSummary"+str(i)+".txt ~/ModelSummary/DateSummary"+str(i)+" 1 B F",shell=True)
	output = float(output)
	output1 = subprocess.check_output("java -cp C_Rouge/C_ROUGE.jar executiverouge.C_ROUGE ~/SysSummary/DateSummary"+str(i)+".txt ~/ModelSummary/DateSummary"+str(i)+" 2 B F",shell=True)
	output1 = float(output1)
	sumR1=sumR1+output
	sumR2=sumR2+output1

avgR1=float(sumR1)/float(7)
avgR2=float(sumR2)/float(7)
print 'Average R1',avgR1
print 'Average R2',avgR2


