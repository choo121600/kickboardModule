import serial
import signal
import threading

from settings import *

line = []

exitThread = False   # 쓰레드 종료용 변수

class Gps:
    def __init__(self):
        self.lon = 0
        self.lat = 0

    #쓰레드 종료용 시그널 함수
    def handler(signum, frame):
        exitThread = True

    #데이터 처리할 함수
    def parsing_data(self, data):
        tmp = ''.join(data)

        data_list = list(tmp.split(','))
        if data_list[0] == '$GPGGA':
            if data_list[3] == 'N':
                self.lat = float(data_list[2][:-7]) + float(data_list[2][-7:]) / 60
            if data_list[5] == 'E':
                self.lon = float(data_list[4][:-7]) + float(data_list[4][-7:]) / 60
        #출력!
        # print(tmp)

    #본 쓰레드
    def readThread(self, ser):
        global line
        global exitThread

        # 쓰레드 종료될때까지 계속 돌림
        while not exitThread:
            #데이터가 있있다면
            for c in ser.read():
                #line 변수에 차곡차곡 추가하여 넣는다.
                line.append(chr(c))

                if c == 10: #라인의 끝을 만나면..
                    #데이터 처리 함수로 호출
                    self.parsing_data(line)
                    speed = str(max_speed)
                    if BOHO[0][0] < self.gps.lat < BOHO[1][0] and BOHO[0][1] < self.gps.lon < BOHO[1][1]:
                        inBoho = 1
                    else:
                        inBoho = 0
                    in_str = "S, " + str(self.lat) + ', ' + str(self.lon) + ', ' + speed + ', ' + str(inBoho) + ", E"
                    ser.write(in_str.encode('utf-8'))             

                    #line 변수 초기화
                    del line[:]                

    def detect_gps(self):
        #종료 시그널 등록
        signal.signal(signal.SIGINT, self.handler)

        #시리얼 열기
        ser = serial.Serial(port, baud, timeout=0)

        #시리얼 읽을 쓰레드 생성
        thread = threading.Thread(target=self.readThread, args=(ser,))

        #시작!
        thread.start() 
