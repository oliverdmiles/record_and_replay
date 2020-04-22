#ifndef VARIABLES
#define VARIABLES

#include <stdint.h>

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 30)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* Enum used to indicate which mode is currently being run */
enum recordReplayPhase { RECORD, REPLAY };

/* Constant set used to determine if an instruction is a synchronization
 * operation */
const std::map<std::string, uint32_t> sync_instrs_to_id = {
    {"BAR", 0}, {"MEMBAR", 1}, {"ATOM", 2},  {"BARRIER", 3},   {"FENCE", 4},
    {"RED", 5}, {"VOTE", 6},   {"MATCH", 7}, {"ACTIVEMASK", 8}};

const std::map<uint32_t, std::string> id_to_sync_instrs = {
    {0, "BAR"}, {1, "MEMBAR"}, {2, "ATOM"},  {3, "BARRIER"},   {4, "FENCE"},
    {5, "RED"}, {6, "VOTE"},   {7, "MATCH"}, {8, "ACTIVEMASK"}};

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int phase = 0;
std::string record_file = "recorded_data.txt";
FILE *fptr;

/* information collected in the instrumentation function */
union Data {
  double d;
  uint64_t u64;
  uint32_t u32[2];
  int64_t i64;
  int32_t i32[2];
};

typedef struct {
  bool is_mem_instr;
  uint32_t time;
  /* bit 31 is type (0 is shared 1 is global)
   * bit 30 is load (0 is store 1 is load)
   * bits 29-0 are thread id
   * would like to record memory operation size and embed as well - probably
   * need 2 bits but maybe 1 */
  uint32_t type_load_tid;
  uint64_t addr;
  Data value;
} record_data;

/* vector to store all accesses for a single kernel call  */
std::vector<record_data> accesses;

/* map from replay files to the next index to be read */
std::map<uint64_t, int> replay_files;

/* number of slots in deviceArr filled by metadata
   slot 0: address with data race
   slot 1: number of threads participating in data race
   slot 2: index of next thread to execute */
__managed__ int NUM_METADATA = 3;

/* array on the device used to replay data races. Format is as follows:
   slots 0-2: metadata. See above
   slot 3x: thread id
   slot 3x + 1: 1 if instruction is a load, 0 otherwise 
   slot 3x + 2: value being loaded or stored */
__managed__ uint64_t **deviceArr;

/* Index of the earlier address by timestamp that still has not resolved all dependencies*/
__device__ int current_laneid = 0;

/* The total number of addresses with data races */
__managed__ uint64_t numDependecies;

/* Thread execution counter */
__device__ uint32_t count = 0;

/* Synchronization variables */
__device__ int mutex = 0;

#endif