
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>

union Data {
    double d;
    uint64_t u64;
    uint32_t u32[2];
    int64_t i64;
    int32_t i32[2];
} data_value;

typedef struct {
    uint32_t tid;
    uint32_t type;
    uint32_t load;
    uint64_t addrs[32];
    Data value;
} record_data;

int main() {
    printf("Hello world!\n");
    record_data temp;
    temp.value.u64 = 0;
    temp.value.u64 -= 1;
    printf("temp is %llu\n", temp.value.d);
    
    return 0;
}