import enum
from PyByteBuffer import ByteBuffer
from typing import List

class TracePacketReader:
    class Type(enum.Enum):
        String = 1
        Binary = 2
    
    class Endianess(enum.Enum):
        Little = 1
        Big = 2
    
    LNK_MRK = 0x0001
    KRN_MRK = 0x0002
    APP_MRK_I32 = 0x0003
    APP_MRK_U32 = 0x0004
    APP_MRK_FLT = 0x0005
    APP_MRK_VAL = 0x0006
    APP_MRK_STR = 0x0007
    APP_MRK_I64 = 0x0008
    APP_MRK_U64 = 0x0009
    PACKET_HEADER_SIZE = 20
    BUFFER_SIZE = 0x100000
    PACKET_HEADER = 0xef56a55a
    
    def __init__(self, inputStream, type, byteOrder, doCheckSum, wordAlign):
        self.doCheckSum = doCheckSum
        self.type = type
        self.inputStream = inputStream
        self.skippedPackets = 0
        self.validPackets = 0
        self.wordAlign = wordAlign

        self.byteOrder = byteOrder
        self.buffer = ByteBuffer.allocate(self.BUFFER_SIZE)
        self.limit = 0
    
    def get_skipped_packets(self):
        return self.skippedPackets
    
    def getValidPackets(self):
        return self.validPackets
    
    def trySyncNextPacketHeader(self):
        if self.checkBufferSize(self.PACKET_HEADER_SIZE):
            nextPosition = self.buffer.position + 1
            
            header = int(self.buffer.get(4, endianness=self.byteOrder))
            if header == self.PACKET_HEADER:
                return True
            else:
                count = 0
                while True:
                    self.buffer.position = nextPosition
                    if self.checkBufferSize(self.PACKET_HEADER_SIZE):
                        nextPosition = self.buffer.position + 1
                        header = int(self.buffer.get(4, endianness=self.byteOrder))
                        if header == self.PACKET_HEADER:
                            return True
                        count += 1
                        if count > 1024 * 64:
                            return False
                    else:
                        break
        return False
    
    def skip_metadata(self):
        for i in range(1024):
            if self.inputStream.read() == 0:
                return
        raise Exception("Could not skip metadata section")
    
    def read_packet(self):
        if self.checkBufferSize(self.PACKET_HEADER_SIZE):
            if self.trySyncNextPacketHeader():
                position = self.buffer.position - 4
                checkSum = 0
                markerType = int(self.buffer.get(2, endianness=self.byteOrder))
                size = int(self.buffer.get(2, endianness=self.byteOrder))
                if self.doCheckSum:
                    for i in range(8):
                        checkSum += self.buffer.get(position + i)
                if markerType == self.LNK_MRK:
                    if self.wordAlign:
                        self.buffer.position = (position + ((size + 3) // 4) * 4)
                    if self.doCheckSum:
                        readCheckSum = int(self.buffer.get(4, endianness=self.byteOrder))
                        if checkSum != readCheckSum:
                            self.skippedPackets += 1
                            raise Exception("Checksum error")
                    return self.TracePacket()
                if self.doCheckSum:
                    for i in range(8, self.PACKET_HEADER_SIZE):
                        checkSum += int(self.buffer.get(position + i))
                eventID = int(self.buffer.get(2, endianness=self.byteOrder))
                eventID = (eventID & 0xFFFF)
                cpuID = int(self.buffer.get(2, endianness=self.byteOrder))
                timeStamp = int(self.buffer.get(8, endianness=self.byteOrder))
                payloadSize = size - self.PACKET_HEADER_SIZE
                if payloadSize < 0:
                    self.skippedPackets += 1
                    raise Exception("Illegal size in trace packet")
                if self.checkBufferSize(payloadSize + (4 if self.doCheckSum else 0)):
                    position = self.buffer.position
                    size -= self.PACKET_HEADER_SIZE
                    eventType = None
                    formats = None
                    fmts = None
                    keys = None
                    if eventID == 0:
                        eventType = ""
                        while True:
                            c = self.buffer.get()
                            if self.doCheckSum:
                                checkSum += c
                            if c == 0:
                                break
                            eventType += chr(c)
                        payloadSize -= len(eventType) + 1
                        if markerType == self.APP_MRK_VAL:
                            format = ""
                            fieldIndex = 0
                            while True:
                                c = self.buffer.get()
                                if self.doCheckSum:
                                    checkSum += c
                                if c == 0:
                                    break
                                format += chr(c)
                            keys = []
                            fmts = []
                            newPayload = ByteBuffer.allocate(payloadSize)
                            formats = format.split("%")
                            key = None
                            for i in range(1, len(formats)):
                                if key == None:
                                    if formats[i].lower() == "s":
                                        key = ""
                                        while True:
                                            c = self.buffer.get()
                                            if self.doCheckSum:
                                                checkSum += c
                                            if c == 0:
                                                break
                                            key += chr(c)
                                        continue
                                if key == None:
                                    key = "field_" + fieldIndex
                                fieldIndex += 1
                                keys.append(key)
                                fmts.append(formats[i])
                                if formats[i].lower() == "s":
                                    while True:
                                        c = self.buffer.get()
                                        if self.doCheckSum:
                                            checkSum += c
                                        newPayload.put(c)
                                        if c == 0:
                                            break
                                elif formats[i].lower() in ["u", "x", "d", "i", "f"]:
                                    for j in range(4):
                                        c = self.buffer.get()
                                        if self.doCheckSum:
                                            checkSum += c
                                        newPayload.put(c)
                                key = None
                            if self.doCheckSum:
                                readCheckSum = self.buffer.get(4,endianness=self.byteOrder)
                                if checkSum != readCheckSum:
                                    self.skippedPackets += 1
                                    self.buffer.position = (position)
                                    raise Exception("Checksum error")
                            if self.wordAlign:
                                self.buffer.position = (position + ((size + 3) // 4) * 4)
                            self.verifyNextHeader(position)
                            self.validPackets += 1
                            return self.TracePacket(timeStamp, markerType, eventType, fmts, keys, cpuID, newPayload)
                    payload = self.buffer.array(payloadSize)
                    if self.doCheckSum:
                        for i in range(len(payload)):
                            checkSum += payload[i]
                    if self.wordAlign:
                        self.buffer.position = (position + ((size + 3) // 4) * 4)
                    if self.doCheckSum:
                        readCheckSum = self.buffer.get(4, endianness=self.byteOrder)
                        if checkSum != readCheckSum:
                            self.skippedPackets += 1
                            raise TracePacketReader.PacketReaderException("Checksum error", True)
                    self.verifyNextHeader(position)
                    self.validPackets += 1
                    return self.TracePacket(timeStamp, markerType, eventID, cpuID, payload)
                else:
                    raise TracePacketReader.PacketReaderException("Trace data stream incomplete", False)
            else:
                raise TracePacketReader.PacketReaderException("No trace packet header detected.", False)
        return None
    
    def verifyNextHeader(self, position):
        if (self.limit - self.buffer.position) >= 4:
            oldPosition = self.buffer.position
            header = int(self.buffer.get(4, endianness=self.byteOrder))
            if header != self.PACKET_HEADER:
                self.skippedPackets += 1
                self.buffer.position = position
                raise TracePacketReader.PacketReaderException("Packet size error", True)
            self.buffer.position = oldPosition
    
    def checkBufferSize(self, neededSize):
        if (self.limit - self.buffer.position) < neededSize:
            if not self.fillBuffer():
                return False
        return (self.limit - self.buffer.position) >= neededSize
    
    def fillBuffer(self):
        remaining = None
        numRemaining = self.limit - self.buffer.position
        if numRemaining > 0:
            remaining = self.buffer.array(numRemaining)
        self.limit = self.BUFFER_SIZE
        self.buffer.rewind()
        if remaining != None:
            self.buffer.put(remaining, size=numRemaining)

        size = self.BUFFER_SIZE - numRemaining
        data = bytearray(size)
        read = self.inputStream.readinto(data)
        if read == -1:
            return False
        self.buffer.rewind()
        self.buffer.put(data, size=read)
        self.buffer.rewind()
        self.limit = read
        return True
    
    class TracePacket:
        def __init__(self, timeStamp: int = 0, markerType: int = 0, eventID: int = 0, cpuID: int = 0, payload: bytearray = None, eventType: str = None, formats: List[str] = None, keys: List[str] = None):
            self.timeStamp = timeStamp
            self.markerType = markerType
            self.event_id = eventID
            self.cpuID = cpuID
            self.payload = payload
            self.eventType = eventType
            self.formats = formats
            self.keys = keys
    
    class PacketReaderException(Exception):
        def __init__(self, message, doContinue):
            super().__init__(message)
            self.doContinue = doContinue
        
        def get_continue(self):
            return self.doContinue
