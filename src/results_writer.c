#define _GNU_SOURCE
#include "results_writer.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

typedef struct
{
    char *name;
    char **values;
    size_t size;
    size_t capacity;
} CsvColumn;

int results_writer_ensure_directory(const char *path)
{
    if (!path || *path == '\0')
        return 0;

    char buffer[PATH_MAX];
    size_t len = strlen(path);
    if (len == 0)
        return 0;
    if (len >= sizeof(buffer))
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(buffer, path, len + 1);

    while (len > 1 && buffer[len - 1] == '/')
    {
        buffer[len - 1] = '\0';
        len--;
    }

    for (char *p = buffer + 1; *p; ++p)
    {
        if (*p == '/')
        {
            *p = '\0';
            if (mkdir(buffer, 0755) != 0 && errno != EEXIST)
            {
                *p = '/';
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(buffer, 0755) != 0 && errno != EEXIST)
        return -1;

    return 0;
}

int results_writer_join_path(char *dest,
                             size_t dest_size,
                             const char *dir,
                             const char *file)
{
    if (!dest || !dir || !file)
    {
        errno = EINVAL;
        return -1;
    }

    size_t dir_len = strlen(dir);
    const char *fmt = (dir_len > 0 && dir[dir_len - 1] != '/') ? "%s/%s" : "%s%s";
    int written = snprintf(dest, dest_size, fmt, dir, file);
    if (written < 0 || (size_t)written >= dest_size)
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    return 0;
}

static const char *fallback_matrix_stem(void)
{
    return "graph";
}

int results_writer_matrix_stem(const char *matrix_path,
                               char *dest,
                               size_t dest_size)
{
    if (!dest || dest_size == 0)
    {
        errno = EINVAL;
        return -1;
    }

    dest[0] = '\0';

    const char *path = matrix_path ? matrix_path : "";
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (!base || *base == '\0')
        base = fallback_matrix_stem();

    size_t base_len = strlen(base);
    if (base_len + 1 > dest_size)
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    memcpy(dest, base, base_len + 1);

    char *dot = strrchr(dest, '.');
    if (dot && dot != dest)
        *dot = '\0';

    if (dest[0] == '\0')
    {
        const char *fallback = fallback_matrix_stem();
        size_t fallback_len = strlen(fallback);
        if (fallback_len + 1 > dest_size)
        {
            errno = ENAMETOOLONG;
            return -1;
        }
        memcpy(dest, fallback, fallback_len + 1);
    }

    return 0;
}

int results_writer_build_results_path(char *dest,
                                      size_t dest_size,
                                      const char *output_dir,
                                      const char *prefix,
                                      const char *matrix_path)
{
    if (!dest || dest_size == 0 || !output_dir || !prefix)
    {
        errno = EINVAL;
        return -1;
    }

    char stem[PATH_MAX];
    if (results_writer_matrix_stem(matrix_path, stem, sizeof(stem)) != 0)
        return -1;

    char filename[PATH_MAX];
    int written = snprintf(filename, sizeof(filename), "%s_%s.csv", prefix, stem);
    if (written < 0 || (size_t)written >= sizeof(filename))
    {
        errno = ENAMETOOLONG;
        return -1;
    }

    return results_writer_join_path(dest, dest_size, output_dir, filename);
}

static void free_column(CsvColumn *col)
{
    if (!col)
        return;
    free(col->name);
    if (col->values)
    {
        for (size_t i = 0; i < col->size; i++)
            free(col->values[i]);
        free(col->values);
    }
    col->name = NULL;
    col->values = NULL;
    col->size = 0;
    col->capacity = 0;
}

static char *clone_string(const char *src)
{
    if (!src)
        return NULL;
    size_t len = strlen(src);
    char *dst = (char *)malloc(len + 1);
    if (!dst)
        return NULL;
    memcpy(dst, src, len + 1);
    return dst;
}

static int append_cell(CsvColumn *col, const char *value)
{
    if (col->size == col->capacity)
    {
        size_t new_capacity = col->capacity == 0 ? 8 : col->capacity * 2;
        char **tmp = (char **)realloc(col->values, new_capacity * sizeof(char *));
        if (!tmp)
            return -1;
        col->values = tmp;
        col->capacity = new_capacity;
    }
    char *copy = clone_string(value ? value : "");
    if (!copy)
        return -1;
    col->values[col->size++] = copy;
    return 0;
}

static int pad_column(CsvColumn *col, size_t target_rows)
{
    while (col->size < target_rows)
    {
        if (append_cell(col, "") != 0)
            return -1;
    }
    return 0;
}

static char *trim_trailing_newline(char *line)
{
    if (!line)
        return line;
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
    {
        line[len - 1] = '\0';
        len--;
    }
    return line;
}

static size_t split_csv_line(char *line, char ***out_fields)
{
    size_t capacity = 8;
    size_t count = 0;
    char **fields = (char **)malloc(capacity * sizeof(char *));
    if (!fields)
    {
        *out_fields = NULL;
        return 0;
    }

    char *cursor = line;
    char *start = line;
    while (1)
    {
        if (*cursor == ',' || *cursor == '\0')
        {
            size_t field_len = (size_t)(cursor - start);
            char *field = (char *)malloc(field_len + 1);
            if (!field)
            {
                for (size_t i = 0; i < count; i++)
                    free(fields[i]);
                free(fields);
                *out_fields = NULL;
                return 0;
            }
            memcpy(field, start, field_len);
            field[field_len] = '\0';
            if (count == capacity)
            {
                size_t new_capacity = capacity * 2;
                char **tmp = (char **)realloc(fields, new_capacity * sizeof(char *));
                if (!tmp)
                {
                    free(field);
                    for (size_t i = 0; i < count; i++)
                        free(fields[i]);
                    free(fields);
                    *out_fields = NULL;
                    return 0;
                }
                fields = tmp;
                capacity = new_capacity;
            }
            fields[count++] = field;
            if (*cursor == '\0')
                break;
            start = cursor + 1;
        }
        cursor++;
    }

    *out_fields = fields;
    return count;
}

static void free_fields(char **fields, size_t count)
{
    if (!fields)
        return;
    for (size_t i = 0; i < count; i++)
        free(fields[i]);
    free(fields);
}

static results_writer_status write_new_file(const char *filename, const char *column_name, const double *values, size_t count)
{
    FILE *out = fopen(filename, "w");
    if (!out)
        return RESULTS_WRITER_IO_ERROR;

    if (fprintf(out, "%s\n", column_name) < 0)
    {
        fclose(out);
        return RESULTS_WRITER_IO_ERROR;
    }

    for (size_t i = 0; i < count; i++)
    {
        if (fprintf(out, "%.6f\n", values[i]) < 0)
        {
            fclose(out);
            return RESULTS_WRITER_IO_ERROR;
        }
    }

    fclose(out);
    return RESULTS_WRITER_OK;
}

results_writer_status append_times_column(const char *filename,
                                          const char *column_name,
                                          const double *values,
                                          size_t count)
{
    if (!filename || !column_name || (!values && count > 0))
        return RESULTS_WRITER_INVALID_ARGS;

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        if (errno == ENOENT)
            return write_new_file(filename, column_name, values, count);
        return RESULTS_WRITER_IO_ERROR;
    }

    CsvColumn *columns = NULL;
    size_t column_count = 0;
    char **header_fields = NULL;
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    results_writer_status status = RESULTS_WRITER_OK;

    line_len = getline(&line, &line_cap, fp);
    if (line_len < 0)
    {
        free(line);
        fclose(fp);
        return write_new_file(filename, column_name, values, count);
    }

    trim_trailing_newline(line);
    size_t header_count = split_csv_line(line, &header_fields);
    if (header_count == 0)
    {
        free(line);
        fclose(fp);
        return write_new_file(filename, column_name, values, count);
    }

    columns = (CsvColumn *)calloc(header_count, sizeof(CsvColumn));
    if (!columns)
    {
        status = RESULTS_WRITER_MEMORY_ERROR;
        goto cleanup;
    }
    column_count = header_count;

    for (size_t i = 0; i < header_count; i++)
    {
        columns[i].name = header_fields[i];
    }
    free(header_fields);
    header_fields = NULL;

    while ((line_len = getline(&line, &line_cap, fp)) >= 0)
    {
        trim_trailing_newline(line);
        char **fields = NULL;
        size_t field_count = split_csv_line(line, &fields);
        if (field_count == 0 && column_count > 0)
        {
            free_fields(fields, field_count);
            continue;
        }
        if (field_count < column_count)
        {
            char **tmp = (char **)realloc(fields, column_count * sizeof(char *));
            if (!tmp)
            {
                free_fields(fields, field_count);
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
            for (size_t j = field_count; j < column_count; j++)
            {
                tmp[j] = clone_string("");
                if (!tmp[j])
                {
                    for (size_t k = field_count; k < j; k++)
                        free(tmp[k]);
                    free(tmp);
                    status = RESULTS_WRITER_MEMORY_ERROR;
                    goto cleanup;
                }
            }
            fields = tmp;
            field_count = column_count;
        }
        for (size_t j = 0; j < column_count; j++)
        {
            if (append_cell(&columns[j], j < field_count ? fields[j] : "") != 0)
            {
                free_fields(fields, field_count);
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
        }
        free_fields(fields, field_count);
    }

    fclose(fp);
    fp = NULL;

    size_t baseline_rows = 0;
    for (size_t i = 0; i < column_count; i++)
        if (columns[i].size > baseline_rows)
            baseline_rows = columns[i].size;

    ssize_t target_idx = -1;
    for (size_t i = 0; i < column_count; i++)
    {
        if (strcmp(columns[i].name, column_name) == 0)
        {
            target_idx = (ssize_t)i;
            break;
        }
    }

    if (target_idx >= 0)
    {
        size_t new_total_rows = baseline_rows + count;
        for (size_t i = 0; i < column_count; i++)
        {
            if (pad_column(&columns[i], baseline_rows) != 0)
            {
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
        }
        for (size_t i = 0; i < column_count; i++)
        {
            if (pad_column(&columns[i], new_total_rows) != 0)
            {
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
        }
        CsvColumn *target = &columns[target_idx];
        size_t start_row = baseline_rows;
        for (size_t i = 0; i < count; i++)
        {
            char buffer[64];
            snprintf(buffer, sizeof(buffer), "%.6f", values[i]);
            char *formatted = clone_string(buffer);
            if (!formatted)
            {
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
            free(target->values[start_row + i]);
            target->values[start_row + i] = formatted;
        }
        baseline_rows = new_total_rows;
    }
    else
    {
        size_t new_total_rows = baseline_rows > count ? baseline_rows : count;
        for (size_t i = 0; i < column_count; i++)
        {
            if (pad_column(&columns[i], new_total_rows) != 0)
            {
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
        }
        CsvColumn *tmp = (CsvColumn *)realloc(columns, (column_count + 1) * sizeof(CsvColumn));
        if (!tmp)
        {
            status = RESULTS_WRITER_MEMORY_ERROR;
            goto cleanup;
        }
        columns = tmp;
        CsvColumn *new_col = &columns[column_count];
        column_count++;
        memset(new_col, 0, sizeof(*new_col));
        new_col->name = clone_string(column_name);
        if (!new_col->name)
        {
            status = RESULTS_WRITER_MEMORY_ERROR;
            goto cleanup;
        }
        for (size_t i = 0; i < new_total_rows; i++)
        {
            if (i < count)
            {
                char buffer[64];
                snprintf(buffer, sizeof(buffer), "%.6f", values[i]);
                if (append_cell(new_col, buffer) != 0)
                {
                    status = RESULTS_WRITER_MEMORY_ERROR;
                    goto cleanup;
                }
            }
            else if (append_cell(new_col, "") != 0)
            {
                status = RESULTS_WRITER_MEMORY_ERROR;
                goto cleanup;
            }
        }
        baseline_rows = new_total_rows;
    }

    fp = fopen(filename, "w");
    if (!fp)
    {
        status = RESULTS_WRITER_IO_ERROR;
        goto cleanup;
    }

    for (size_t i = 0; i < column_count; i++)
    {
        if (fprintf(fp, "%s%s", columns[i].name, (i + 1 < column_count) ? "," : "") < 0)
        {
            status = RESULTS_WRITER_IO_ERROR;
            goto cleanup;
        }
    }
    if (fprintf(fp, "\n") < 0)
    {
        status = RESULTS_WRITER_IO_ERROR;
        goto cleanup;
    }

    for (size_t row = 0; row < baseline_rows; row++)
    {
        for (size_t col = 0; col < column_count; col++)
        {
            const char *cell = row < columns[col].size ? columns[col].values[row] : "";
            if (fprintf(fp, "%s%s", cell ? cell : "", (col + 1 < column_count) ? "," : "") < 0)
            {
                status = RESULTS_WRITER_IO_ERROR;
                goto cleanup;
            }
        }
        if (fprintf(fp, "\n") < 0)
        {
            status = RESULTS_WRITER_IO_ERROR;
            goto cleanup;
        }
    }

cleanup:
    if (fp)
        fclose(fp);
    free(line);
    if (columns)
    {
        for (size_t i = 0; i < column_count; i++)
            free_column(&columns[i]);
        free(columns);
    }
    free(header_fields);

    return status;
}
