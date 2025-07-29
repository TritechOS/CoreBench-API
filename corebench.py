#Built-in packages
import time
import os
import socket
import multiprocessing
import threading
import math
import platform
import random
from datetime import datetime
import subprocess
import re
import getpass
import csv
import sys
import shutil
import hashlib
#Third-party packages
import cpuinfo
import psutil
import GPUtil
import distro
import speedtest
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests

#Custom packages

def is_connected():
    try:
        response = requests.get("https://corebench.me", timeout=5)
        if response.status_code == 200:
            return True
        
    except requests.ConnectionError:
        return False


def return_api_key():
    if os.path.exists("apikey.txt"):
        with open("apikey.txt", "r") as f:
            key = f.read().strip()
        return key
    else:
        key = None
        return key

def write_api_key(key):
    with open("apikey.txt", "w") as f:
        f.write(key)

def get_file_hash():
    sha256 = hashlib.sha256()
    filename = os.path.abspath(__file__)

    with open(filename, 'rb') as f:
        file_bytes = f.read()

    file_text = file_bytes.decode('utf-8')
    normalized_text = file_text.replace('\r\n', '\n').rstrip('\n')
    sha256.update(normalized_text.encode('utf-8'))

    return sha256.hexdigest()

def sendForAuth(cpu_name, core_count, thread_count, ram, single, mcore, mthread, gflops, fullLoad, full, os_name, version, key):
    server_ip = "https://submit.corebench.me/submit"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":"application/json"
    }
    data = {
        "payload":
        {
            "cpu_name": cpu_name,
            "core_count":core_count,
            "thread_count":thread_count,
            "ram":ram,
            "single_core": single,
            "multi_core": mcore,
            "multi_thread": mthread,
            "gflops": gflops,
            "full_load": fullLoad,
            "overall_score": full,
            "os_name": os_name,
            "version": version
        },

        "timestamp": time.time(),

        "signature":get_file_hash(),
        "purpose":""
    }

    response = requests.post(server_ip, json=data, headers=headers)    
    return response

def apiCheck(apiKey):
    server_ip = "https://submit.corebench.me/submit"

    headers = {
        "Authorization": f"Bearer {apiKey}",
        "Content-Type":"application/json"
    }
    data = {
        "payload":
        {
            "cpu_name": "",
            "core_count":"",
            "thread_count":"",
            "ram":"",
            "single_core": "",
            "multi_core": "",
            "multi_thread": "",
            "gflops": "",
            "full_load": "",
            "overall_score": "",
            "os_name": "",
            "version": ""
        },

        "timestamp": time.time(),

        "signature":get_file_hash(),

        "purpose":"API_CHECK"
    }
    
    response = requests.post(server_ip, json=data, headers=headers)    
    return response

def upload_and_return_status(name, core_count, thread_count, memory, score, mcore, mthread, gflops, fullLoad, final, distro, version, key):
            try:
                response = sendForAuth(name, core_count, thread_count, memory, score, mcore, mthread, gflops, fullLoad, final, distro, version, key)
                status_code = response.status_code
                message = response.text

                if status_code:
                    return status_code

            except Exception as e:
                return 500



def get_user():
    return getpass.getuser()

def getData():
    try:
    
        global hostname, GPUs, osName, architecture, brandName, clockSpeed, \
            systemCoreCount, memRaw, memory, endLoad, distroName, localIp, \
            Threads, threadsPerCore, osNamePretty, user, version #improved readability
        
        try:
            hostname = socket.gethostname()
            localIp = socket.gethostbyname(socket.gethostname())
            GPUs = GPUtil.getGPUs()
            osName = platform.system()
            architecture = platform.machine()
            brandName = cpuinfo.get_cpu_info()["brand_raw"]
            
            def get_advertised_cpu_clock():
                try:
                    output = subprocess.check_output("lscpu", shell=True, text=True)
                    for line in output.split("\n"):
                        if "CPU max MHz" in line:  # Find the max advertised MHz
                            max_mhz = float(line.split(":")[1].strip().split()[0])  # Extract and convert to float
                            return f"{max_mhz/1000:.2f} GHz"  # Convert MHz to GHz and round to 2 decimal places
                except Exception as e:
                    return str(e)
            
            clockSpeed = get_advertised_cpu_clock()
            systemCoreCount = psutil.cpu_count(logical=False)
            Threads = os.cpu_count()
            threadsPerCore= int(os.cpu_count())/int(systemCoreCount)
            memRaw = round(((psutil.virtual_memory().total)/(1024**2)))
            memory = math.ceil(((psutil.virtual_memory().total)/(1024**3)))

            user = get_user()
            if osName in ["Linux"]:
                time.sleep(1)
                distroName = str(distro.name(pretty=True))

                def distroColour():
                    result = subprocess.run(["neofetch"], capture_output=True, text=True)
        
                    f=open("NeofetchOut.txt", "w")
                    f.write(result.stdout)
                    f.close()
        
                    file_path = "NeofetchOut.txt"
                    f=open("NeofetchOut.txt", "r")
                    contents=f.read()
                    pattern = r"\033\[(3[0-7]|9[0-7])m"
                    match = re.findall(pattern, contents)
        
                    while "37" in match or "97" in match:
                        i=-1
                        for item in match:
                            i+=1
                            if item.strip() == "97" or item.strip() == "37":
                                match.pop(i)
        
        
                    distroColourCode = f"\033[{match[0]}m"
                    return distroColourCode
                osNamePretty=f"{distroColour()}{distroName}"
                os.remove("NeofetchOut.txt")
            
            else:
                if osName.lower() in ["nt", "dos", "windows"]:
                    osNamePretty=osName
                else:
                    osNamePretty=osName
                
        except Exception as e:
            quit()
            
        #UPDATE THIS WITH EVERY VERSION
        version = "API"
        #UPDATE THIS WITH EVERY VERSION
        
        endLoad = True
    #the classic messages we always have, feel free to add, trying to keep this one less bloated
    except Exception as e:
        f = open("log.txt","w")
        f.write(str(e))
        f.close()

    return {"hostname":hostname, "GPUs":GPUs, "osName":osName,
            "architecture":architecture, "brandName":brandName, "clockSpeed":clockSpeed,
            "coreCount":systemCoreCount, "memRaw":memRaw, "memory":memory, "distroName":distroName, "localIP":localIp,
            "threadCount":Threads, "threadsPerCore":threadsPerCore, "osNamePretty":osNamePretty, 
            "username":user, "version":version}


N = 1000000 #a bit of a magic number


def setSingleCoreAffinity():
    p = psutil.Process(os.getpid())
    
    validCore = False
    try: #try
        p.cpu_affinity([0])
        validCore = True
    except: #try harder
        for item in p.cpu_affinity():
            if validCore == True:
                break
            else:
                try:
                    p.cpu_affinity([item])
                    validCore = True
                except:
                    p.cpu_affinity(list(range(os.cpu_count()))) #reset

    if validCore == False:
        quit() #give up


def calculateGFLOPS(stageNo, coreCount):
    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(os.cpu_count())))

    matrixSize = 1024 #number of iterations

    matrixA = np.random.rand(matrixSize, matrixSize)
    matrixB = np.random.rand(matrixSize, matrixSize)

    start = time.perf_counter()

    oldPercentageComplete = -1
    iterations = 5000 #number of iterations

    for _ in range(3):
        resultantMatrix = np.dot(matrixA,matrixB) # warmup avoiding CPU frequency scaling issues

    recordedGFLOPS = []

    iterationFLOPS = 0
    iterationGFLOPS = 0

    for iterationNo in range(iterations):
        startTime = time.perf_counter_ns()

        resultantMatrix = np.dot(matrixA, matrixB)  # Matrix multiplication

        endTemp = time.perf_counter_ns()

        iterationFLOPS = (2 * matrixSize**3) / (endTemp/1000000000 - startTime/1000000000)
        iterationGFLOPS = iterationFLOPS/1000000000


        recordedGFLOPS.append(iterationGFLOPS)

    end = time.perf_counter()

    #compute the mean
    averageGFLOPS = sum(recordedGFLOPS)/len(recordedGFLOPS)

    #calculate the standard deviation from the mean
    stdDeviationNumerator = 0

    for dataPoint in recordedGFLOPS:
        value = (dataPoint-averageGFLOPS)**2
        stdDeviationNumerator += value
    
    stdDeviation = math.sqrt(stdDeviationNumerator/len(recordedGFLOPS))

    #calculate the Z-score for each point
    normalisedData = []
    discarded = []
    for dataPoint in recordedGFLOPS:
        zScore = (dataPoint-averageGFLOPS)/(stdDeviation)

        if zScore < 2 and zScore > -2:
            normalisedData.append(dataPoint)
        else:
            discarded.append(dataPoint)
            pass # <-- too anomalous, discarded

    averageGFLOPS = sum(normalisedData)/len(normalisedData)
    percentDiscarded = (len(discarded)/len(recordedGFLOPS))*100

    #totalTime = end-start --- REMOVED DUE TO OBSOLESCENSE
    #avgTime = totalTime/6
    
    #score = round((1/(avgTime/(3*math.e)))*(math.e)*(1000*(1/math.log(coreCount+6,10))))

    #timeList.append(totalTime)
    #scoreList.append(score)

    FLOPS = (2 * matrixSize**3) / ((end - start)/iterations)
    
    GFLOPS = averageGFLOPS

    cpuFrequency = psutil.cpu_freq().current * 1e6
    operationsPerCycle = (GFLOPS * 1e9) / cpuFrequency
    

    return round(GFLOPS,2)

def singleCore():
    global systemCoreCount, fullTest

    setSingleCoreAffinity()

    testCoreCount = 6

    scoreList = []
    timeList = []

    percentageComplete = 0

    ballHeight = 5000 #metres
    
    gravitationalEntropy = random.randint(-10,10)/10
    acceleration = 9.81+gravitationalEntropy
    bounceConstant = random.randint(1, 10)

    timeSimulated = 0
    timeIncrement = 1e-6
    distanceTravelled = 0


    timeUntilCollision = math.sqrt(ballHeight/(0.5*acceleration))

    ticker = -1

    timeList = []

    start = time.perf_counter()

    for x in range(0,3):

        ticker += 1
        yVel = 0
        timeSimulated = 0
        oldPercentageComplete = -1

        roundStart = time.perf_counter()
        while distanceTravelled < ballHeight:


            yVel = yVel-timeIncrement*acceleration
            timeSimulated+=timeIncrement

            
            distanceTravelled = 0.5 * acceleration * timeSimulated**2

        yVel = -yVel - (bounceConstant)

        timeList.append(timeSimulated)

        u = yVel
        a = acceleration

        distanceTravelled = 0
        estimatedHeight =  (u**2)/(2*a)

    end = time.perf_counter()

    totalTime = end-start
    avgTime = (end-start)/3

    score = round((1/(avgTime/(3*math.e)))*(math.e)*(1000*(1/math.log(testCoreCount+4,10))))

    allPassTimeAvg = sum(timeList)/3
    allPassTimeAvg = float(str(allPassTimeAvg).rstrip("0").rstrip("."))

    #Stage 1 algorithm end#

    percentageAccuracy = 100.0 - (((allPassTimeAvg-timeUntilCollision)/timeUntilCollision) *100.0)

    scoreList.append(score)
    timeList.append(totalTime)
    
    time.sleep(3)

    #Stage 2

    percentageComplete = 0

    arrowHeight = 5000
    
    gravitationalEntropy = random.randint(-10,10)/10
    acceleration = 9.81+gravitationalEntropy

    timeSimulated = 0
    timeIncrement = 1e-6

    yDistanceTravelled = 0
    xDistanceTravelled = 0

    yVel = 0
    xVel = 50

    timeUntilCollision = math.sqrt(arrowHeight/(0.5*acceleration))

    ticker = -1

    timeList = []
    oldPercentageComplete = -1

    start = time.perf_counter()

    for x in range(0,3):

        ticker+=1
        yVel = 0
        yDistanceTravelled = 0
        xDistanceTravelled = 0
        timeSimulated = 0

        roundStart = time.perf_counter()
        while yDistanceTravelled < arrowHeight:

            yVel = yVel - timeIncrement*acceleration

            timeSimulated+=timeIncrement
            yDistanceTravelled = 0.5 * acceleration * timeSimulated**2

            xDistanceTravelled = xVel*timeSimulated

            angle = -math.atan2(yVel,xVel)*(180/math.pi) #radians to degrees
            resultantVelocity = math.sqrt(yVel**2+xVel**2) #calculates the resultant velocity
    end = time.perf_counter()

    totalTime = end-start
    avgTime = (end-start)/3
    score = round((1/(avgTime/(3*math.e)))*(math.e)*(1000*(1/math.log(testCoreCount+4,10))))

    scoreList.append(score)
    timeList.append(totalTime)

    time.sleep(3)


    score = int(round(sum(scoreList)/2))
    totalTime = sum(timeList)

    return score

def get_physical_core_ids():
    output = subprocess.check_output(['lscpu', '-p=CPU,Core,Socket'], text=True)
    lines = output.strip().split('\n')
    
    physical_cores = {}
    for line in lines:
        if line.startswith('#'):
            continue
        cpu_id_str, core_id_str, socket_id_str = line.strip().split(',')
        cpu_id = int(cpu_id_str)
        core_id = int(core_id_str)
        socket_id = int(socket_id_str)

        key = (core_id, socket_id)
        if key not in physical_cores:
            physical_cores[key] = cpu_id 
    
    return sorted(physical_cores.values())

#Full Load Test SUBROUTINES
def fp_benchmark(data_chunk):
    total = 0

    for i in data_chunk:
        if i == 0:
            continue #funny bug lol

        total += math.sin(i) * math.cos(i) + \
                 math.log(i + 1) * math.sqrt(i) + \
                 math.exp(i % 10) + \
                 math.factorial(i % 10) + \
                 math.tan(i / 3) * math.atan(i / 2) + \
                 math.pow(i, 2) + \
                 math.sqrt(math.fabs(math.sin(i))) + \
                 math.log(math.fabs(math.cos(i) + 1)) + \
                 math.sin(i * 2) * math.cos(i * 2)

    return total

def full_load_benchmark(func, data, num_cores, iterations=50):
    times = []
    chunk_size = len(data) // num_cores

    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]


    for _ in range(iterations):
        start_time = time.perf_counter()

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(func, chunks)

        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return sum(times) / len(times)

def run_full_load_benchmark(num_cores, data):
    chunk_size = len(data) // num_cores
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_cores)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        avg_time = full_load_benchmark(fp_benchmark, data, num_cores)
        return avg_time

def full_load_intermission(gflops=0):
    time.sleep(3)

    data = list(range(5_000_000))
    num_cores = multiprocessing.cpu_count()
    avg_time = run_full_load_benchmark(num_cores, data)

    score = round((2 / avg_time) * 1000 / math.log(avg_time + math.e))

    time.sleep(3)

    return score

#Full Load Test SUBROUTINES END

def multiCore():     
    def intense1(threadNo, coreID):
        p = psutil.Process(os.getpid())
        p.cpu_affinity(coreID)

        arrowHeight = 500
        
        gravitationalEntropy = random.randint(-10,10)/10
        ACCELERATION = 9.81+gravitationalEntropy

        timeSimulated = 0
        timeIncrement = 1e-6

        yDistanceTravelled = 0
        xDistanceTravelled = 0

        yVel = 0
        xVel = 50

        timeToHit = math.sqrt(arrowHeight/(0.5*ACCELERATION))

        while yDistanceTravelled < arrowHeight:

            yVel = yVel - timeIncrement*ACCELERATION

            timeSimulated+=timeIncrement
            yDistanceTravelled = 0.5 * ACCELERATION * timeSimulated**2

            xDistanceTravelled = xVel*timeSimulated

            angle = -math.atan2(yVel,xVel)*(180/math.pi) #radians to degrees
            resultantVelocity = math.sqrt(yVel**2+xVel**2) #calculates the resultant velocity
    

    coreCount = 6

    #running process function (creates variable numbers of functions in dynamic mode)
    
    timeList = []

    def run_processes():
        global testCoreCount, systemCoreCount, coreContext

        testCoreCount = 12

        processes = []

        #validation process
        logical = os.cpu_count()
        coreList = get_physical_core_ids()

        # if any core ID is invalid (>= logical), fallback
        if any(core >= logical for core in coreList):
            coreList = list(range(systemCoreCount))

        for i in range(testCoreCount//2):
            coreRun = coreList[i%len(coreList)]

            p = multiprocessing.Process(target=intense1, args=(i + 1, [coreRun]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
                    

    for x in range(0,3):
        start=time.perf_counter()

        if __name__ == "__main__":     
            run_processes()

        end=time.perf_counter()
        Time = end-start
        timeList.append(Time)

    totalTime = 0
    for item in timeList:
        totalTime+=item

    avgTime = totalTime/3
    score = round((1/(avgTime/(math.e/1.8))*(math.e)*(1000*(1/math.log(coreCount-4,10))))/2)

    gflops = calculateGFLOPS(stageNo="2", coreCount=coreCount)

    time.sleep(3)

    full_load_score = full_load_intermission(gflops=gflops)

        

    return score, gflops, full_load_score


def multiThread(showResults):
    logical_cores = os.cpu_count()
    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(logical_cores)))

    def intense1(procNo, timeList, core_id, core_show, thread_id, thread_pass):
        p = psutil.Process(os.getpid())
        p.cpu_affinity([core_id])

        ballHeight = 250
        acceleration = 9.81 + random.randint(-10, 10) / 10
        ballConstant = random.randint(1, 10)
        timeSimulated = 0
        timeIncrement = 1e-6
        distanceTravelled = 0
        yVel = 0
        while distanceTravelled < ballHeight:
            yVel -= timeIncrement * acceleration
            timeSimulated += timeIncrement
            distanceTravelled = 0.5 * acceleration * timeSimulated**2
        yVel = -yVel - ballConstant
        timeList.append(timeSimulated)

    def intense2(procNo, timeList, core_id, core_show, thread_id, thread_pass):
        p = psutil.Process(os.getpid())
        p.cpu_affinity([core_id])

        arrowHeight = 250
        ACCELERATION = 9.81 + random.randint(-10, 10) / 10
        timeSimulated = 0
        timeIncrement = 1e-6
        yDistanceTravelled = 0
        xVel = 50
        yVel = 0
        while yDistanceTravelled < arrowHeight:
            yVel -= timeIncrement * ACCELERATION
            timeSimulated += timeIncrement
            yDistanceTravelled = 0.5 * ACCELERATION * timeSimulated**2
            xDistanceTravelled = xVel * timeSimulated
            angle = -math.atan2(yVel, xVel) * (180 / math.pi)
            resultantVelocity = math.sqrt(yVel**2 + xVel**2)
        timeList.append(timeSimulated)

    threadCount = 12 

    def run_processes():
        with multiprocessing.Manager() as manager:
            timeList = manager.list()
            processes = []

            total_logical_threads = os.cpu_count()
            logical_thread_ids = list(range(total_logical_threads))

            thread_pass = 0

            threadCount = 12

            for i in range(threadCount):
                #wrap around logical threads if there are fewer than 12
                logical_id = logical_thread_ids[i % total_logical_threads]

                core_show = logical_id // 2  #just for display
                thread_id = logical_id % 2
                proc_no = i + 1
                thread_pass = i // total_logical_threads

                if proc_no % 2 == 0:
                    p = multiprocessing.Process(target=intense2, args=(proc_no, timeList, logical_id, core_show, thread_id, thread_pass))
                else:
                    p = multiprocessing.Process(target=intense1, args=(proc_no, timeList, logical_id, core_show, thread_id, thread_pass))

                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            return list(timeList)


    timeResults = []

    for _ in range(3):
        start = time.perf_counter()
        timeList = run_processes()
        end = time.perf_counter()
        timeResults.append(end - start)

    totalTime = sum(timeResults)
    avgTime = totalTime / 3

    score = round((1/(avgTime/(math.e/1.5))*(math.e)*(1000*(1/math.log(threadCount-6,10)))))

    return score



def fullCPUTest():
    global fullTest, brandName, version, distroName, systemCoreCount, Threads

    fullTest = True

    def coolDown():
        time.sleep(13)
        
    singleCoreScore = singleCore()
    coolDown()
    multiCoreScore, gflops, fullLoadScore = multiCore()
    coolDown()
    multiThreadScore = multiThread()
    
    totalScore = singleCoreScore+multiCoreScore+multiThreadScore

    finalScore = int(round(totalScore/3))

    return brandName, systemCoreCount, Threads, memRaw, singleCoreScore, multiCoreScore, multiThreadScore, gflops, fullLoadScore, finalScore, distroName, version, apikey

def test_speed():
    try:
        for x in range(0,10):
            try:
                st = speedtest.Speedtest()
                connected=True
                break
            except:
                connected=False
                pass
                
        if not connected:
            st = speedtest.Speedtest()
                
        downloads = []
        uploads = []
        pings = []
        
        st.get_best_server()
        
        for x in range(0,3):
            download = st.download()/1e+6
            downloads.append(download)
    
            upload = st.upload()/1e+6
            uploads.append(upload)

            ping = st.results.ping
            pings.append(ping)
        
        download = round(sum(downloads)/3,2)
        upload = round(sum(uploads)/3,2)
        ping = round(sum(pings)/3,2)
        

        score = int(round((((download+(upload*15)))/2)-ping))
        
    except speedtest.ConfigRetrievalError as e:
        f=open("error.txt","w")
        f.write(e)
        f.close()
        download=0
        upload=0
        ping=0
        score=0

    return download, upload, ping, score


def get_latest_release(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data['tag_name'], response.status_code
    elif response.status_code == 404:
        return "No releases found or repository does not exist.", response.status_code
    else:
        return f"Error: {response.status_code}", response.status_code

def check_is_latest_version(version):
    versionTag = version.replace(".", "")

    latestTag, error = get_latest_release("TriTechX", "corebench")
    latestVersion = latestTag.replace("CoreBench", "")

    if versionTag == latestVersion:
        return True, ".".join(latestVersion), error
    else:
        return False, ".".join(latestVersion), error
    

'''
sc - single core
st - single core
mc - multi core
mt - multi thread
nic - internet speed
n - internet speed
api - set api key
'''

def run_test(test_command):
    fullTest = False
    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(os.cpu_count())))

    validChoice = ["sc", "st", "mc", "mt", "nic", "n", "fullc", "fc", "api", "home"]
    otherChoice = ["exit", "quit", "clear"]
    validArgs = ["d"]
    valid = False

    args = None
    multiArgs = False
    skip = False
    
    selection = test_command
    selection = selection.lower().strip(" ")

    if "*" in selection:
        choice, num = selection.split("*")[0].strip(), selection.split("*")[1].strip()
        
        try:
            num = int(num.strip())
        except:
            num = int(re.sub(r"-\s*d", "", num.strip()))
    else:
        choice = selection.split("-")[0].strip() if "-" in selection else selection
        num = 1  # Default to 1 if no *x syntax is used

            
    try:
        # Checks to see if it contains args
        try:
            args = selection.split(" -")[1]
        except:
            args = selection.split("-")[1]
        finally:
            pass
            
        multiArgs = True

        if choice in validChoice:
            skip = False
        else:
            skip = True
    except:
        # Has no args, gets sent here
        # If the choice is valid, it goes through the next validation process     
        if choice in validChoice:
            valid = True
            skip = False
        else:
            valid = False
            skip = True
    
    if not skip:
        if not multiArgs:
            dynamicMode = False
            valid = True
            # Checks for correct base but no args

        elif args == "d":
            dynamicMode = True
            valid = True
            # Dynamic mode
            
        else:
            dynamicMode = False
            valid = False
            # No valid arguments

    else:
        valid = False
        # Invalid base

    base = choice
    index = -1  # Initialize index with a default value
    
    if base in ["sc", "st"]:
        index = 0
    elif base == "mc":
        index = 1
    elif base == "mt":
        index = 2
    elif base in ["fullc", "fc"]:
        index = 3
    elif base in ["n", "nic"]:
        index = 4
    elif base == "api":
        index = 5
    elif base == "home":
        index = 6
    
    if index == 0:
        for x in range(num):
            singleCore()
        
    elif index == 1:
        for x in range(num):
            multiCore()
        
    elif index == 2:
        for x in range(num):
            multiThread()
        
    elif index == 3:
        for x in range(num):
            fullCPUTest()
        
    elif index == 4:
        for x in range(num):
            test_speed()
        
    elif index == 5:
        request_api_key()

    elif index == 6:
        showHome()
    else:
        pass
