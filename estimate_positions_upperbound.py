from math import factorial as fa

a1 = (fa(25) // (fa(5) * fa(5) * fa(15))) * 15
a2 = (fa(25) // (fa(5) * fa(4) * fa(16))) * 16 * 2
a3 = (fa(25) // (fa(4) * fa(4) * fa(17))) * 17

print(a1+a2+a3)
