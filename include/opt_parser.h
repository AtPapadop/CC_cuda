#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dynamic integer list used by CLI parsing helpers.
typedef struct
{
    int *values;
    size_t size;
    size_t capacity;
} OptIntList;

// Initialize an empty OptIntList with no heap allocations.
void opt_int_list_init(OptIntList *list);

// Release memory held by an OptIntList and reset its fields.
void opt_int_list_free(OptIntList *list);

// Append a single integer to the list, growing capacity on demand. Returns 0 on success.
int opt_int_list_append(OptIntList *list, int value);

// Sort the list in ascending order and remove duplicate values in-place.
void opt_int_list_sort_unique(OptIntList *list);

// Parse a decimal string into a positive integer. Returns 0 on success.
int opt_parse_positive_int(const char *text, int *out);

// Parse comma/range specifications (e.g., "1,4,8" or "1:16:2") into a list. Returns 0 on success.
int opt_parse_range_list(const char *spec, OptIntList *list, const char *label);

#ifdef __cplusplus
}
#endif
