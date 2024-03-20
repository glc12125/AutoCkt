import os
from collections import defaultdict
import argparse
from eval_engines.ae.nucleus_parser.trace_packet_reader import TracePacketReader
from enum import Enum
from typing import List
import numpy as np


CPU_IDLE_PERCENTAGE_METRIC_NAME = "idle_percentage"
CPU_TASK_MAX_UTILIZATION_METRIC_NAME = "cpu_task_max_utilization"

class FieldType(Enum):
    UTF8 = 1
    UINT = 2
    SINT = 3
    FLOAT = 4

class Field:
    def __init__(self, name: str, field_type: FieldType, length: int, field_id: str):
        self.name = name
        self.field_type = field_type
        self.length = length
        self.field_id = field_id

class Fields:
    def __init__(self):
        self.fields = []

    def add_field(self, field: Field):
        self.fields.append(field)

class Event:
    def __init__(self, name: str):
        self.name = name
        self.fields = Fields()

    def set_fields(self, fields: Fields):
        self.fields = fields

class EventType:
    def __init__(self, event: Event, field_labels: List[str]):
        self.event = event
        self.field_labels = field_labels

class EventTypeModel:
    def __init__(self, event_type: EventType, event_model: Event):
        self.event_type = event_type
        self.event_model = event_model

class TmfTimestamp:
    def __init__(self, timestamp: int):
        self.timestamp = timestamp

class TmfEventSource:
    def __init__(self, id: str):
        self.id = id

class ByteBuffer:
    def __init__(self, capacity: int):
        self.buffer = bytearray(capacity)
        self.position = 0
        self.limit = capacity

    def rewind(self):
        self.position = 0

    def limit(self, limit: int):
        self.limit = limit

    def put_int(self, value: int):
        self.buffer[self.position:self.position+4] = value.to_bytes(4, 'big')
        self.position += 4

    def put_long(self, value: int):
        self.buffer[self.position:self.position+8] = value.to_bytes(8, 'big')
        self.position += 8

    def put(self, value: bytes):
        self.buffer[self.position:self.position+len(value)] = value
        self.position += len(value)

    def flip(self):
        self.limit = self.position
        self.position = 0

class TmfEvent:
    def __init__(self, timestamp: TmfTimestamp, source: TmfEventSource, event_type: EventType, event_ref):
        self.timestamp = timestamp
        self.source = source
        self.event_type = event_type
        self.event_ref = event_ref
        self.content = None

    def set_content(self, content):
        self.content = content

class EventContent:
    def __init__(self, event: TmfEvent, content, payload: ByteBuffer):
        self.event = event
        self.content = content
        self.payload = payload

    def parse_content(self):
        pass

class CPUState:
    def __init__(self, cpu_id: int):
        self.cpu_id = cpu_id
        self.cpu_state = -1
        self.last_idle_time = -1

class CPUUtil:
    def __init__(self, cpu_id: int, initialTime: float):
        self.cpu_id = cpu_id
        self.initial_time = initialTime
        self.idle_entry_time = 0
        self.idle_time = 0

class TraceDataImporter:
    EVENT_DEFINITION_ID = "eventDefId"
    EVENT_TOOLCHAIN_ID = "toolchain"
    EVENT_SYNC_RECIEVE_ID = "syncRecieve"
    EVENT_SYNC_SEND_ID = "syncSend"
    DEFAULT_EVENT_DEF_ID = "EventDefinition.Nucleus.Kernel.4_0"
    MIN_PACKET_SIZE = 20
    CURRENT_UTC_TIME_EVENT_ID = 152
    BINARY_DATA = 0
    STRING_DATA = 1
    PACKET_ALIGNMENT = "packetAlignment"
    TRUE_STR = "true"
    FALSE_STR = "false"
    BYTE_ORDERING = "endianess"
    LITTLE_END_STR = "little"
    BIG_END_STR = "big"
    EFF_USER_TRACE = "userTrace.eff"
    EFF_KERNEL_TRACE = "kernelTrace.eff"
    STATEDUMP_TASKS_NAME = "Tasks"
    STATEDUMP_TASKS_TYPE = 65536
    EVENT_BUFFER_SIZE = 1024

    def __init__(self, rawLogFile, messageStream, outputFolder, byteOrdering, dataType, hasMetadata, validateCheckSum, toolchainName, simulation_time, start_cpu_stats_time):
        self.rawLogFile = rawLogFile
        self.messageStream = messageStream
        self.outputFolder = outputFolder
        self.effKernelTraceFileName = None
        self.effUserTraceFileName = None
        self.effSyncFileName = None
        self.byteOrdering = byteOrdering
        self.dataType = dataType
        self.hasMetadata = hasMetadata
        self.validateCheckSum = validateCheckSum
        self.clearTraceFile = None
        self.syncEventName = None
        self.syncEventsData = defaultdict(list)
        self.toolchainName = toolchainName
        self.event_type_lut = {}
        self.usedEventTypes = set()
        self.not_found_event_types = set()
        self.ustEventTypeLUT = {}
        self.eventSourceLUT = {}
        self.overflowEventTypeModel = None
        #self.eventRef = TmfEventReference()
        self.hasKernelTrace = False
        self.hasUserTrace = False
        self.markBufferOverflow = False
        self.stateDump = None
        self.skippedPackets = None
        self.eventDefinitionId = self.DEFAULT_EVENT_DEF_ID
        self.wordAlign = False
        self.kernelEvents = None
        self.userEvents = None
        self.syncUtcTime = None
        self.simulation_time = simulation_time
        # stats member vars
        self.active_cpus = {}
        self.active_cpu_busy_percentage = {}
        self.observation_interval = 10 # default observation interval in msecs
        self.active_cpu_states = {}
        self.active_cpu_idle_durations = {}
        self.start_cpu_stats_indexes = {}
        self.start_cpu_stats_time = start_cpu_stats_time
        self.enable_test = False


    def setOutputFolderPath(self, outputFolder):
        self.outputFolder = outputFolder

    def getOutputFolderPath(self):
        return self.outputFolder

    def setInputFile(self, rawLogFile):
        self.rawLogFile = rawLogFile

    def getInputFile(self):
        return self.rawLogFile

    def set_byte_order(self, byteOrdering):
        self.byteOrdering = byteOrdering

    def getByteOrder(self):
        return self.byteOrdering

    def setValidateCheckSum(self, validataChkSum):
        self.validateCheckSum = validataChkSum

    def isValidateCheckSum(self):
        return self.validateCheckSum

    def getDataType(self):
        return self.dataType

    def setDataType(self, dataType):
        self.dataType = dataType

    def setMessageStream(self, messageStream):
        self.messageStream = messageStream

    def setMarkBufferOverflow(self):
        self.markBufferOverflow = True

    def isHasMetadata(self):
        return self.hasMetadata

    def setHasMetadata(self, hasMetadat):
        self.hasMetadata = hasMetadat

    def getSkippedPackets(self):
        return self.skippedPackets

    def getNotFoundIDs(self):
        return list(self.notFoundEventTypes)

    def getEffKernelTraceFile(self):
        file = os.path.join(self.effKernelTraceFileName)
        if self.hasKernelTrace:
            return file
        else:
            if os.path.exists(file):
                os.remove(file)
            return None

    def getEffUserTraceFile(self):
        file = os.path.join(self.effUserTraceFileName)
        if self.hasUserTrace:
            return file
        else:
            if os.path.exists(file):
                os.remove(file)
            return None

    def getEffKernelTraceFileName(self):
        return self.effKernelTraceFileName

    def setEffKernelTraceFileName(self, effKernelTraceFileName):
        self.effKernelTraceFileName = effKernelTraceFileName

    def getEffUserTraceFileName(self):
        return self.effUserTraceFileName

    def setEffUserTraceFileName(self, effUserTraceFileName):
        self.effUserTraceFileName = effUserTraceFileName

    def getSyncFileName(self):
        return self.effSyncFileName

    def setSyncFileName(self, syncFileName):
        self.effSyncFileName = syncFileName

    def isClearTraceFile(self):
        return self.clearTraceFile

    def setClearTraceFile(self, clearTraceFile):
        self.clearTraceFile = clearTraceFile

    def getSyncEventName(self):
        return self.syncEventName

    def setSyncEventName(self, syncEventName):
        self.syncEventName = syncEventName

    def getToolchainName(self):
        return self.toolchainName

    def setToolchainName(self, toolchainName):
        self.toolchainName = toolchainName

    def setStateDump(self, stateDump):
        self.stateDump = stateDump

    def getWordAlign(self):
        return self.wordAlign

    def setWordAlign(self, wordAlign):
        self.wordAlign = wordAlign

    def parse_trace_file(self) -> TmfTimestamp:
        timestamp = TmfTimestamp(0)
        raw_log_input_stream = None
        packet_reader = None
        try:
            raw_log_input_stream = open(self.rawLogFile, 'rb')
            packet_reader = TracePacketReader(raw_log_input_stream, self.dataType if self.dataType == TracePacketReader.Type.String else TracePacketReader.Type.Binary, ( "little" if self.byteOrdering == TracePacketReader.Endianess.Little else "big"), self.validateCheckSum, self.wordAlign)
            if self.hasMetadata:
                try:
                    packet_reader.skip_metadata()
                except Exception as e:
                    raise Exception("Error while processing metadata: " + str(e))
            events_result = []
            if self.enable_test:
                gt_file = open('/media/liangchuan/ZX1/development/SM/AE_automation/RL/nucleus_parser/data/ECU1_SoC1_CortexA53_nucleus_gt_events.txt', 'r')
                lines = gt_file.readlines()
                total_count = len(lines)
                matched_count = 0
            index = 0
            while True:
                try:
                    packet = packet_reader.read_packet()
                    if packet is None:
                        break
                    cpu_id = packet.cpuID
                    cpu_util_inst = self.active_cpus.get(cpu_id, None)
                    cpu_state_inst = self.active_cpu_states.get(cpu_id, None)
                    event_id = packet.event_id
                    if event_id in [16, 17]:
                        events_result.append("{},{},{},{}".format(packet.timeStamp, packet.markerType, packet.event_id, packet.cpuID))
                        if self.enable_test:
                            if events_result[index] == lines[index].strip():
                                matched_count += 1
                            else:
                                print("Do not match! index: {}, packet: {}, gt: {}".format(index, events_result[index], lines[index]))
                        index += 1
                        if self.enable_test:
                            if index == total_count:
                                assert matched_count == total_count

                        if cpu_util_inst is None:
                            initial_time = packet.timeStamp
                            cpu_util_inst = CPUUtil(cpu_id, initial_time)
                            self.active_cpus[cpu_id] = cpu_util_inst

                        if event_id == 16:
                            idle_entry_time = packet.timeStamp
                            cpu_util_inst.idle_entry_time = idle_entry_time

                            # updates for states
                            if cpu_state_inst is None:
                                cpu_state_inst = CPUState(cpu_id)
                                cpu_state_inst.last_idle_time = idle_entry_time
                                self.active_cpu_states[cpu_id] = cpu_state_inst

                            if cpu_state_inst.cpu_state != 0:
                                cpu_state_inst.cpu_state = 0
                                cpu_state_inst.last_idle_time = idle_entry_time
                        if event_id == 17:
                            idle_exit_time = packet.timeStamp
                            if cpu_util_inst.idle_entry_time != 0 and idle_exit_time != 0:
                                cpu_util_inst.idle_time += idle_exit_time - cpu_util_inst.idle_entry_time
                            
                            # updates for states
                            if cpu_state_inst is None:
                                cpu_state_inst = CPUState(cpu_id)
                                self.active_cpu_states[cpu_id] = cpu_state_inst

                            if cpu_state_inst.cpu_state != 1:
                                cpu_state_inst.cpu_state = 1
                            
                            # calculate idle durations
                                
                            idle_durations = self.active_cpu_idle_durations.get(cpu_id, None)

                            if idle_durations is None:
                                self.active_cpu_idle_durations[cpu_id] = []
                            
                            if cpu_state_inst.last_idle_time != -1:
                                idle_duration = (idle_exit_time - cpu_state_inst.last_idle_time) / 1000000
                                self.active_cpu_idle_durations[cpu_id].append(idle_duration)

                    if cpu_util_inst is not None:
                        final_time = packet.timeStamp
                        time_interval = (final_time - cpu_util_inst.initial_time) / 1000000
                        if time_interval >= self.observation_interval:
                            busy_percentage = (time_interval - cpu_util_inst.idle_time / 1000000) / time_interval
                            if busy_percentage >= 0 and busy_percentage <= 1:
                                busy_percentages = self.active_cpu_busy_percentage.get(cpu_id, None)
                                if cpu_id not in self.start_cpu_stats_indexes:
                                    self.start_cpu_stats_indexes.update({cpu_id: -1})
                                if self.start_cpu_stats_indexes[cpu_id] == -1 and packet.timeStamp >= self.start_cpu_stats_time:
                                    print("setting start_cpu_stats_index: {}".format(len(self.active_cpu_busy_percentage[cpu_id])))
                                    self.start_cpu_stats_indexes[cpu_id] = len(self.active_cpu_busy_percentage[cpu_id])
                                if busy_percentages is None:
                                    busy_percentages = []
                                    busy_percentages.append(busy_percentage)
                                    self.active_cpu_busy_percentage[cpu_id] = busy_percentages
                                else:
                                    busy_percentages.append(busy_percentage)
                            cpu_util_inst.initial_time = final_time
                            cpu_util_inst.idle_time = 0
                    
                    event_type = None
                    if packet.markerType == TracePacketReader.LNK_MRK:
                        continue
                    elif packet.markerType == TracePacketReader.KRN_MRK:
                        if packet.event_id in self.event_type_lut:
                            etm = self.event_type_lut[packet.event_id]
                            event_type = etm.type
                            if packet.event_id not in self.used_event_types:
                                self.used_event_types.append(packet.event_id)
                        else:
                            self.not_found_event_types.add(packet.event_id)
                    elif packet.markerType in [TracePacketReader.APP_MRK_I32, TracePacketReader.APP_MRK_U32, TracePacketReader.APP_MRK_I64, TracePacketReader.APP_MRK_U64, TracePacketReader.APP_MRK_FLT, TracePacketReader.APP_MRK_STR, TracePacketReader.APP_MRK_VAL]:
                        event_type = self.get_event_type(packet)
                    else:
                        raise Exception("Unknown trace marker found: " + str(packet.markerType))
                    if event_type is None:
                        raise TracePacketReader.PacketReaderException("Event type not found.", True)
                    timestamp = TmfTimestamp(packet.timestamp)
                    source = self.get_event_source("cpu_" + str(packet.cpu_id))
                    payload = None
                    if packet.payload is not None:
                        payload = ByteBuffer(len(packet.payload))
                        payload.order(self.byte_ordering)
                        payload.put(packet.payload)
                        payload.flip()
                    if packet.event_id == TraceDataImporter.CURRENT_UTC_TIME_EVENT_ID:
                        self.sync_utc_time = TmfTimestamp(payload.get_long() * 1000000000)
                        payload.rewind()
                    ev = TmfEvent(timestamp, source, event_type, self.event_ref)
                    if payload is not None:
                        ev.set_content(EventContent(ev, None, payload))
                    if self.mark_buffer_overflow:
                        self.mark_buffer_overflow = False
                        overflow_event = TmfEvent(timestamp, self.get_event_source("kernelawareness"), self.overflow_event_type_model.type, self.event_ref)
                        overflow_payload = ByteBuffer(4)
                        overflow_payload.order(self.byte_ordering)
                        overflow_payload.limit(4)
                        overflow_payload.put_int(0)
                        overflow_payload.flip()
                        overflow_event.set_content(EventContent(ev, None, overflow_payload))
                        print("Overflow event written: {}".format(overflow_event))
                        self.has_kernel_trace = True
                    try:
                        if packet.markerType == TracePacketReader.KRN_MRK:
                            print("Kernel event written: {}".format(ev))
                            self.has_kernel_trace = True
                        else:
                            print("user event written: {}".format(ev))
                            self.has_user_trace = True
                    except Exception as e:
                        print("Error: {}".format(e))
                except TracePacketReader.PacketReaderException as e1:
                    if not e1.get_continue():
                        raise Exception("Error while parsing trace data: " + str(e1))
                except Exception as e:
                    raise Exception("Error while parsing trace data: " + str(e))
            print("CPU Utilization stats:")
            for cpu_id, percentages in self.active_cpu_busy_percentage.items():
                info = "CPU {}: [".format(cpu_id)
                for percentage in percentages:
                    info += ",{:.2f}".format(percentage * 100)
                info += "]\n"
                print(info)
            print("CPU State stats:")
            for cpu_id, durations in self.active_cpu_idle_durations.items():
                info = "CPU {}: [".format(cpu_id)
                total_idle_time = 0.0
                simulation_time = 700.0
                for duration in durations:
                    total_idle_time += duration
                busy_time = simulation_time - total_idle_time
                info += "total idle time: {:.2f} msecs, idle percentage: {:.2f}%, total busy time: {:.2f} msecs, busy percentage: {:.2f}%\n".format(total_idle_time, total_idle_time / simulation_time * 100, busy_time, busy_time / simulation_time * 100)
                print(info)
        finally:
            if raw_log_input_stream is not None:
                raw_log_input_stream.close()
            skipped_packets = packet_reader.get_skipped_packets()
        return timestamp


    def get_event_type(self, packet: TracePacketReader.TracePacket) -> EventType:
        type = packet.get_event_type()
        if type in self.ust_event_type_lut:
            return self.ust_event_type_lut[type]
        else:
            marker_type = packet.get_marker_type()
            event_config = Event(type)
            fields = Fields()
            event_config.set_fields(fields)
            if marker_type == TracePacketReader.APP_MRK_VAL:
                format = packet.get_formats()
                keys = packet.get_keys()
                for i in range(len(format)):
                    field = Field(keys[i], FieldType.UTF8 if format[i].lower() == "s" else FieldType.UINT if format[i].lower() == "u" else FieldType.SINT if format[i].lower() in ["d", "i"] else FieldType.FLOAT, 0 if format[i].lower() == "s" else 32, "EventField.String" if format[i].lower() == "s" else "EventField.Long" if format[i].lower() in ["u", "x"] else "EventField.Long" if format[i].lower() in ["d", "i"] else "EventField.Double")
                    fields.add_field(field)
            else:
                field = Field("field_0", FieldType.SINT if marker_type in [TracePacketReader.APP_MRK_I32, TracePacketReader.APP_MRK_I64] else FieldType.UINT if marker_type in [TracePacketReader.APP_MRK_U32, TracePacketReader.APP_MRK_U64] else FieldType.FLOAT if marker_type == TracePacketReader.APP_MRK_FLT else FieldType.UTF8, 32 if marker_type in [TracePacketReader.APP_MRK_I32, TracePacketReader.APP_MRK_U32] else 64 if marker_type in [TracePacketReader.APP_MRK_I64, TracePacketReader.APP_MRK_U64] else 32 if marker_type == TracePacketReader.APP_MRK_FLT else 0, "EventField.Long" if marker_type in [TracePacketReader.APP_MRK_I32, TracePacketReader.APP_MRK_U32, TracePacketReader.APP_MRK_I64, TracePacketReader.APP_MRK_U64] else "EventField.Double" if marker_type == TracePacketReader.APP_MRK_FLT else "EventField.String")
                fields.add_field(field)
            event_type = EventType(event_config, ["value"])
            if type not in self.event_type_lut:
                self.event_type_lut[type] = EventTypeModel(event_type, event_config)
            else:
                self.message_stream.write(("Duplicate event ID (" + str(type) + ") found in loaded event-metadata.\n").encode())
            return event_type

    def get_event_source(self, id: str) -> TmfEventSource:
        if id in self.event_source_lut:
            return self.event_source_lut[id]
        else:
            source = TmfEventSource(id)
            self.event_source_lut[id] = source
            return source

    def has_sync_file(self) -> bool:
        return self.sync_utc_time is not None
    
    def get_outliers(self, data, q1_ratio=25, q2_ratio=75, threshold_ratio=1.5) -> list:
        np_data = np.array(data)
        q1 = np.percentile(np_data, q1_ratio)
        q3 = np.percentile(np_data, q2_ratio)
        print("q1: {}".format(q1))
        print("q3: {}".format(q3))
        iqr = q3 - q1
        print("iqr: {}".format(iqr))
        threshold = threshold_ratio * iqr
        print("threshold: {}".format(threshold))
        outliers = np.where((np_data <= q1 - threshold) | (np_data >= q3 + threshold))
        return np_data[outliers]


    def get_cpu_stats(self) -> dict:
        specs = {
            CPU_IDLE_PERCENTAGE_METRIC_NAME: [],
            CPU_TASK_MAX_UTILIZATION_METRIC_NAME: []
        }

        print("self.start_cpu_stats_indexes:")
        for k, v in self.start_cpu_stats_indexes.items():
            print("CPU " + str(k) + "'s stats index: " + str(v))
        # Calculate CPU utilization metrics
        for cpu_id, percentages in self.active_cpu_busy_percentage.items():
            for idx, percentage in enumerate(percentages):
                if idx < self.start_cpu_stats_indexes[cpu_id]:
                    continue
                specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME].append(percentage)
        # Calculate CPU states metrics
        for cpu_id, durations in self.active_cpu_idle_durations.items():
            total_idle_time = 0.0
            for duration in durations:
                total_idle_time += duration
            specs[CPU_IDLE_PERCENTAGE_METRIC_NAME].append(total_idle_time / self.simulation_time)

        #outliers = self.get_outliers(specs[CPU_MAX_UTILIZATION_METRIC_NAME], q1_ratio=15, q2_ratio=75, threshold_ratio=1.5)
        #print("Outliers1 of array is : \n", outliers)
        #specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME] = np.max(specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME])
        #specs[CPU_IDLE_PERCENTAGE_METRIC_NAME] = np.mean(specs[CPU_IDLE_PERCENTAGE_METRIC_NAME])
        if len(specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME]) > 0:
            specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME] = np.max(specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME])
        else:
            del specs[CPU_TASK_MAX_UTILIZATION_METRIC_NAME]
        if len(specs[CPU_IDLE_PERCENTAGE_METRIC_NAME]) > 0:
            specs[CPU_IDLE_PERCENTAGE_METRIC_NAME] = np.mean(specs[CPU_IDLE_PERCENTAGE_METRIC_NAME])
        else:
            del specs[CPU_IDLE_PERCENTAGE_METRIC_NAME]
        
        return specs

class SyncEventData:
    def __init__(self, timestamp: int, priority: int, correlation_id: int, name: str):
        self.timestamp = timestamp
        self.priority = priority
        self.correlation_id = correlation_id
        self.name = name


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--trace_file', type=str, default="/media/liangchuan/ZX1/development/SM/AE_automation/RL/nucleus_parser/data/ECU1_SoC1_CortexA53_nucleus.trace")
  args = parser.parse_args()
  
  task_first_start_time = 113070574

  importer = TraceDataImporter(args.trace_file, None, "output", TracePacketReader.Endianess.Little, TracePacketReader.Type.Binary, False, False, "toolchain", 700.0, task_first_start_time)
  importer.parse_trace_file()
  specs = importer.get_cpu_stats()
  print("specs: {}".format(specs))

if __name__=="__main__":
  main()

