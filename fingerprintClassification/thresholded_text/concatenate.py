import os

filesA = [f for f in os.listdir('A') if f[-4:] == '.txt']
filesL = [f for f in os.listdir('L') if f[-4:] == '.txt']
filesR = [f for f in os.listdir('R') if f[-4:] == '.txt']
filesT = [f for f in os.listdir('T') if f[-4:] == '.txt']
filesW = [f for f in os.listdir('W') if f[-4:] == '.txt']

current = 'W'

if current == 'A':
    current_files = filesA
elif current == 'L':
    current_files = filesL
elif current == 'R':
    current_files = filesR
elif current == 'T':
    current_files = filesT
elif current == 'W':
    current_files = filesW

with open('{}.txt'.format(current), 'w') as outfile:
    for fname in current_files:
        with open('{}/{}'.format(current, fname), 'r') as infile:
            outfile.write(infile.read())

print('{} concatenation done!'.format(current))
