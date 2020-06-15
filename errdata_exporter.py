def exportError():
    firstTime = True
    epochBreaker = 0
    epochCount = 1
    with open("errors.txt", "r") as f:
        lines = f.readlines()
    with open("errors.txt", "w") as f:
        for line in lines:
            if 'x is:' in line:
                if epochCount == 1001:
                    break

                if firstTime == True:
                    f.write("Epoch %i.\n" % epochCount)
                    epochCount = epochCount + 1
                    firstTime = False

                if epochBreaker == 0:
                    line = line.replace("x is: ", "\tHidden Layer Errors: ")
                elif epochBreaker == 1:
                    line = line.replace("x is: ", "\tOutput Layer Errors: ")
                elif epochBreaker == 2:
                    f.write("\nEpoch %i.\n" % epochCount)
                    epochCount = epochCount + 1
                    epochBreaker = 0
                    line = line.replace("x is: ", "\tHidden Layer Errors: ")
                f.write(line)
                epochBreaker = epochBreaker+1
if __name__ == '__main__':
    exportError()