from math import factorial as fa

# 25マスに手番側のコマ(5|4)個、相手のコマ(5|4)個、残りは空白を置く場合の数（カッコの中）に、空白のどこに穴を置くかの場合の数を最後に乗算した。
a1 = (fa(25) // (fa(5) * fa(5) * fa(15))) * 15
a2 = (fa(25) // (fa(5) * fa(4) * fa(16))) * 16
a3 = (fa(25) // (fa(4) * fa(4) * fa(17))) * 17

print(a1)
print(a2)
print(a3)
print(a1 + a2 * 2 + a3)


# 25マスのうち、上三角の15マスのどこかに穴を置いてから、残りの24マスに手番側のコマ(5|4)個、相手のコマ(5|4)個、空白を置く　と考えた。
# key observation: 上三角に含まれない10マスに穴を置いた場合については、残りの24マスの配置をどう選んでも、その結果の盤面に対して対称性のある盤面が、列挙した盤面のなかに必ず含まれる。
b1 = 15 * (fa(24) // (fa(5) * fa(5) * fa(14)))
b2 = 15 * (fa(24) // (fa(5) * fa(4) * fa(15)))
b3 = 15 * (fa(24) // (fa(4) * fa(4) * fa(16)))

print(b1)
print(b2)
print(b3)
print(b1 + b2 * 2 + b3)

#ooo--
#-oo--
#--o--
#-----
#-----
# 25マスのうち、上図でoになっている6マスのどこかに穴を置いてから、残りの24マスに手番側のコマ(5|4)個、相手のコマ(5|4)個、空白を置く　と考えた。
# key observation: 上図で-になっている19マスのどこかに穴を置いた場合については、残りの24マスの配置をどう選んでも、その結果の盤面に対して対称性のある盤面が、列挙した盤面のなかに必ず含まれる。
c1 = 6 * (fa(24) // (fa(5) * fa(5) * fa(14)))
c2 = 6 * (fa(24) // (fa(5) * fa(4) * fa(15)))
c3 = 6 * (fa(24) // (fa(4) * fa(4) * fa(16)))

print(c1)
print(c2)
print(c3)
print(c1 + c2 * 2 + c3)
