from models import get_basic_model
import numpy
from pphelper import c2i
from pphelper import i2c
from pphelper import ALPHABET
import sys
import random

seq_length = 100

if len(sys.argv) < 3:
    print("Usage: python3 generate.py <seed-file> <weights-file>")

seedfile = sys.argv[1]
weightsfile = sys.argv[2]
with open(seedfile) as f:
    s = f.read()
s = ''.join(list(filter(lambda x: x in ALPHABET, s)))
    

print("vectorizing seed...")
dataX = []
dataY = []
for i in range(len(s) - seq_length):
    seq_in = s[i:i + seq_length]
    seq_out = s[i + seq_length]
    dataX.append([c2i(char) for char in seq_in])
    dataY.append(c2i(seq_out))

print("building model...")
model = get_basic_model()
model.load_weights(weightsfile)
start = numpy.random.randint(0, len(dataX)-1)
      
print("making predictions...")
pattern = dataX[start]
last_pred = None
num_reps = 0
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(ALPHABET))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    if random.randrange(0,2) == 1:
        prediction = numpy.delete(prediction, index)
        index = numpy.argmax(prediction)
    result = i2c(index)
    if result == last_pred:
        num_reps += 1
    if num_reps > 5:
        del prediction[index]
        index = numpy.argmax(prediction)
        result = i2c(index)
    seq_in = [i2c(value) for value in pattern]
    print(result, end="")
    sys.stdout.flush()
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
