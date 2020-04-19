#example inputs
inputs = [1,2,3,2.5]

#example weights
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

#example bias
bias = [2, 3, 0.5]

#length of inputs and weights to be used in the for loop
m = len(weights)
n = len(inputs)

#dummy variables
inner = []
output = []
outer = 0


for i in range(m):
    for j in range(n):
        #multiply inputs by the weight
        inner = inputs[j]*weights[i][j]
        #sum each contribution
        outer = inner + outer
    #add the bias term
    outer = outer + bias[i]
    #append result to a list of outputs    
    output.append(outer)
    #reset dummy variable
    outer = 0    

print(output)



