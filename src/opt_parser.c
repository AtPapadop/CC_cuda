#define _GNU_SOURCE
#include "opt_parser.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void opt_int_list_init(OptIntList *list)
{
    if (!list)
        return;
    list->values = NULL;
    list->size = 0;
    list->capacity = 0;
}

void opt_int_list_free(OptIntList *list)
{
    if (!list)
        return;
    free(list->values);
    list->values = NULL;
    list->size = 0;
    list->capacity = 0;
}

int opt_int_list_append(OptIntList *list, int value)
{
    if (!list || value <= 0)
        return -1;
    if (list->size == list->capacity)
    {
        size_t new_capacity = list->capacity == 0 ? 8 : list->capacity * 2;
        int *tmp = (int *)realloc(list->values, new_capacity * sizeof(int));
        if (!tmp)
            return -1;
        list->values = tmp;
        list->capacity = new_capacity;
    }
    list->values[list->size++] = value;
    return 0;
}

static int cmp_int(const void *a, const void *b)
{
    int lhs = *(const int *)a;
    int rhs = *(const int *)b;
    return (lhs > rhs) - (lhs < rhs);
}

void opt_int_list_sort_unique(OptIntList *list)
{
    if (!list || list->size == 0)
        return;
    qsort(list->values, list->size, sizeof(int), cmp_int);
    size_t write = 1;
    for (size_t read = 1; read < list->size; read++)
    {
        if (list->values[read] != list->values[write - 1])
            list->values[write++] = list->values[read];
    }
    list->size = write;
}

int opt_parse_positive_int(const char *text, int *out)
{
    if (!text || *text == '\0' || !out)
        return -1;

    errno = 0;
    char *endptr = NULL;
    long value = strtol(text, &endptr, 10);
    if (errno != 0 || endptr == text || *endptr != '\0' || value <= 0 || value > INT_MAX)
        return -1;

    *out = (int)value;
    return 0;
}

static char *trim(char *token)
{
    if (!token)
        return token;
    while (isspace((unsigned char)*token))
        token++;
    size_t len = strlen(token);
    while (len > 0 && isspace((unsigned char)token[len - 1]))
        token[--len] = '\0';
    return token;
}

static int parse_range_token(const char *token, OptIntList *list)
{
    char *spec = strdup(token);
    if (!spec)
        return -1;

    char *first_colon = strchr(spec, ':');
    if (!first_colon)
    {
        int single = 0;
        int rc = opt_parse_positive_int(spec, &single);
        if (rc != 0 || opt_int_list_append(list, single) != 0)
        {
            free(spec);
            return -1;
        }
        free(spec);
        return 0;
    }

    *first_colon = '\0';
    char *second_colon = strchr(first_colon + 1, ':');
    if (second_colon)
        *second_colon = '\0';

    int start = 0;
    int end = 0;
    int step = 1;

    if (opt_parse_positive_int(spec, &start) != 0 ||
        opt_parse_positive_int(first_colon + 1, &end) != 0 ||
        (second_colon && opt_parse_positive_int(second_colon + 1, &step) != 0))
    {
        free(spec);
        return -1;
    }

    if (step <= 0 || start > end)
    {
        free(spec);
        return -1;
    }

    for (long long current = start; current <= end; current += step)
    {
        if (current > INT_MAX)
        {
            free(spec);
            return -1;
        }
        if (opt_int_list_append(list, (int)current) != 0)
        {
            free(spec);
            return -1;
        }
        if (current > (long long)INT_MAX - step)
            break;
    }

    free(spec);
    return 0;
}

int opt_parse_range_list(const char *spec, OptIntList *list, const char *label)
{
    if (!spec || !list)
        return -1;

    char *copy = strdup(spec);
    if (!copy)
        return -1;

    char *saveptr = NULL;
    char *token = strtok_r(copy, ",", &saveptr);
    if (!token)
    {
        fprintf(stderr, "Invalid %s specification: '%s'\n", label ? label : "value", spec);
        free(copy);
        return -1;
    }

    int rc = 0;
    do
    {
        char *trimmed = trim(token);
        if (*trimmed == '\0' || parse_range_token(trimmed, list) != 0)
        {
            fprintf(stderr, "Invalid %s specification near '%s'\n", label ? label : "value", trimmed);
            rc = -1;
            break;
        }
    } while ((token = strtok_r(NULL, ",", &saveptr)) != NULL);

    free(copy);

    if (rc != 0)
        return -1;

    if (list->size == 0)
    {
        fprintf(stderr, "No %s provided.\n", label ? label : "values");
        return -1;
    }

    opt_int_list_sort_unique(list);
    return 0;
}
