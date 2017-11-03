# -*- coding: cp1252 -*-
from nltk.tree import *
from nltk.draw import tree

print Tree(1, [2, 3, 4])
##(1 2 3 4)
s = Tree('S', [Tree('NP', ['I']),Tree('VP', [Tree('V', ['saw']),Tree('NP', ['him'])])])
print s
s.draw()
##(S (NP I) (VP (V saw) (NP him)))


dp1 = Tree('dp', [Tree('d', ['the']), Tree('np', ['dog'])])
dp2 = Tree('dp', [Tree('d', ['the']), Tree('np', ['cat'])])
vp = Tree('vp', [Tree('v', ['chased']), dp2])
vp = Tree('vp', [Tree('v', ['chased']), dp2])
sentence = Tree('s', [dp1, vp])
print sentence
sentence.draw()
