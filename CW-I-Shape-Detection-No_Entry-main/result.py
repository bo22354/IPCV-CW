import shutil
import subprocess


fileName = 'No_entry/NoEntry'

for x in range(16):
    newFileName = fileName + str(x) + '.bmp'
    command = ["python", "main.py", "-n", newFileName]
    result = subprocess.run(command, capture_output=True, text=True)
    print(newFileName)
    src1 = 'circles.jpg'
    src2 = 'detected.jpg'
    src3 = 'violaJones.jpg'
    src4 = 'allBoxes.jpg'
    dst1 = 'Results/circles' + str(x) + '.jpg'
    dst2 = 'Results/detected' + str(x) + '.jpg'
    dst3 = 'Results/violaJones' + str(x) + '.jpg'
    dst4 = 'Results/allBoxes' + str(x) + '.jpg'

    shutil.copy(src1, dst1)
    shutil.copy(src2, dst2)
    shutil.copy(src3, dst3)
    shutil.copy(src4, dst4)

    print(result.stdout)




# Run the command and capture output