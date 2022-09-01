
LQIDD_TP = "../result/4504/true_positives.txt"
LQIDD_TN = "../result/4504/true_negatives.txt"
Jawdah_TP = "../result/Jawdah_TP_split.txt"
Jawdah_FN = "../result/Jawdah_FN_split.txt"
Jawdah_FP = "../result/Jawdah_FP_split.txt"

f1 = open(LQIDD_TP,'r')
LQIDD_TPs = f1.readlines()

f2 = open(Jawdah_TP,'r')
Jawdah_TPs = f2.readlines()

f3 = open(Jawdah_FN,'r')
Jawdah_FNs = f3.readlines()

f4 = open(LQIDD_TN,'r')
LQIDD_TNs = f4.readlines()

f5 = open(Jawdah_FP,'r')
Jawdah_FPs = f5.readlines()


TPinTP = []
for name in Jawdah_TPs:
    if name in LQIDD_TPs:
        TPinTP.append(name.strip())
print(TPinTP)
print(len(TPinTP))
print(len(Jawdah_TPs))

LQIDD_Jawdah = []
for name in LQIDD_TPs:
    if name not in Jawdah_TPs:
        LQIDD_Jawdah.append(name.strip())
print(LQIDD_Jawdah)
print(len(LQIDD_Jawdah))
print(len(Jawdah_TPs))

Jawdah_LQIDD = []
for name in Jawdah_TPs:
    if name not in LQIDD_TPs:
        Jawdah_LQIDD.append(name.strip())
print(Jawdah_LQIDD)
print(len(Jawdah_LQIDD))
print(len(Jawdah_TPs))

print(len(LQIDD_TPs))

# FNinTP = []
# for name in Jawdah_FNs:
#     if name in LQIDD_TPs:
#         FNinTP.append(name.strip())
# print(FNinTP)
# print(len(FNinTP))
# print(len(Jawdah_FNs))
# print(FNinTP)

FPinTN = []
for name in Jawdah_FPs:
    if name in LQIDD_TNs:
        FPinTN.append(name.strip())
print(FPinTN)
print(len(FPinTN))
print(len(Jawdah_FPs))

