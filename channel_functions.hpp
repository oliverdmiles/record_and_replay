#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/channel.hpp"

extern std::vector<record_data> accesses;


/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
__managed__ ChannelDev channel_dev;
ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* Function used to indicate that there are no more values being added to the
 * channel */
__global__ void flush_channel() {
    /* push memory access with negative clock value to communicate the kernel is
    * completed */
    record_data rd;
    rd.time = UINT32_MAX;
    channel_dev.push(&rd, sizeof(record_data));

    /* flush channel */
    channel_dev.flush();
}

/* Function used to receive data from the device on the host */
void *recv_thread_fun(void *) {
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;
        if (recv_thread_receiving && (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {

            uint32_t num_processed_bytes = 0;

            while (num_processed_bytes < num_recv_bytes) {
                record_data *rd = (record_data *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                */
                if (rd->time == UINT32_MAX) {
                recv_thread_receiving = false;
                break;
                }

                accesses.push_back(*rd);
                num_processed_bytes += sizeof(record_data);
            }
        }
    }
    free(recv_buffer);
    return NULL;
}