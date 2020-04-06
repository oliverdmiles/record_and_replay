#ifndef VARIABLES
#define VARIABLES

#include <stdint.h>


/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
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
std::string filename = "recorded_data.txt";
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
  uint64_t addr[32];
  Data value[32];
} record_data;

/* vector to store all accesses for a single kernel call  */
std::vector<record_data> accesses;

/* array to access on memory */
__managed__ int **deviceArr;
__managed__ int start = 0;
__managed__ int NUM_METADATA = 3;

__device__ uint32_t count = 0;
__device__ int mutex = 0;

#endif