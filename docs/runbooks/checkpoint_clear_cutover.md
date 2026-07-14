# LangGraph Checkpoint Clear Cutover

This runbook replaces checkpoint schema/node migration. The approved cutover
policy is to preserve one database backup, stop every writer, clear the three
LangGraph checkpoint data tables, and start the new graph with an empty
conversation state.

Do not run these commands while a host backend, Docker `app` service, worker,
or test process can write checkpoints. Do not clear `checkpoint_migrations`;
it records the PostgreSQL saver schema version rather than user sessions.

## 1. Resolve the running PostgreSQL container without printing credentials

```powershell
$container = docker compose ps -q postgres
if ([string]::IsNullOrWhiteSpace($container)) {
    throw "PostgreSQL container is not running"
}

$containerEnv = docker inspect $container --format '{{json .Config.Env}}' |
    ConvertFrom-Json
$databaseEnv = @{}
foreach ($entry in $containerEnv) {
    $parts = $entry -split '=', 2
    $databaseEnv[$parts[0]] = $parts[1]
}
$databaseUser = $databaseEnv['POSTGRES_USER']
$databaseName = $databaseEnv['POSTGRES_DB']
if ([string]::IsNullOrWhiteSpace($databaseUser) -or
    [string]::IsNullOrWhiteSpace($databaseName)) {
    throw "PostgreSQL container identity is incomplete"
}
```

Keep these variables in the current process. Do not echo them together with a
password and do not write a database URI into this repository.

## 2. Stop writers and capture before-counts

Stop the host backend and any worker manually. If the Compose app is running:

```powershell
docker compose stop app
```

Then record counts without checkpoint bodies:

```powershell
$countSql = @"
SELECT 'checkpoint_blobs', count(*) FROM checkpoint_blobs
UNION ALL SELECT 'checkpoint_writes', count(*) FROM checkpoint_writes
UNION ALL SELECT 'checkpoints', count(*) FROM checkpoints
UNION ALL SELECT 'checkpoint_migrations', count(*) FROM checkpoint_migrations
ORDER BY 1;
"@
docker exec $container psql -v ON_ERROR_STOP=1 -U $databaseUser `
    -d $databaseName -At -c $countSql
```

If counts change after writers were stopped, find and stop the remaining
writer before continuing.

## 3. Create and verify a custom-format backup

The backup lives outside the repository. Use an absolute destination directory
that already exists and is access-controlled.

```powershell
$backupDirectory = $env:A3_CHECKPOINT_BACKUP_DIR
if ([string]::IsNullOrWhiteSpace($backupDirectory) -or
    -not (Test-Path -LiteralPath $backupDirectory -PathType Container)) {
    throw "Set A3_CHECKPOINT_BACKUP_DIR to an existing protected directory"
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$containerBackup = "/tmp/a3-checkpoints-$stamp.dump"
$hostBackup = Join-Path $backupDirectory "a3-checkpoints-$stamp.dump"

docker exec $container pg_dump -v -U $databaseUser -d $databaseName `
    --format=custom --file=$containerBackup `
    --table=public.checkpoints `
    --table=public.checkpoint_blobs `
    --table=public.checkpoint_writes
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed" }

docker exec $container pg_restore --list $containerBackup | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pg_restore verification failed" }

docker cp "${container}:$containerBackup" $hostBackup
if ($LASTEXITCODE -ne 0 -or
    -not (Test-Path -LiteralPath $hostBackup -PathType Leaf) -or
    (Get-Item -LiteralPath $hostBackup).Length -eq 0) {
    throw "Checkpoint backup copy is missing or empty"
}
docker exec $container rm -- $containerBackup
```

Do not continue until the backup path, size, and `pg_restore --list` result are
recorded in the cutover report. Never commit the dump.

## 4. Clear only checkpoint data tables

```powershell
$clearSql = @"
BEGIN;
LOCK TABLE checkpoint_writes, checkpoint_blobs, checkpoints
    IN ACCESS EXCLUSIVE MODE;
TRUNCATE TABLE checkpoint_writes, checkpoint_blobs, checkpoints;
COMMIT;
"@
docker exec $container psql -v ON_ERROR_STOP=1 -U $databaseUser `
    -d $databaseName -c $clearSql
if ($LASTEXITCODE -ne 0) { throw "Checkpoint clear failed" }
```

Do not add `checkpoint_migrations` to this statement. Do not use `CASCADE` and
do not truncate every table in the database.

## 5. Verify zero legacy state and restart

Run the count query again. `checkpoints`, `checkpoint_blobs`, and
`checkpoint_writes` must all be zero; `checkpoint_migrations` must remain
non-zero. Start only the new graph/backend, issue one controlled smoke request,
and verify that:

1. one new thread can write and reload a checkpoint;
2. the checkpoint graph/runtime fingerprint matches the new served graph;
3. no old node ID or legacy terminal schema is present;
4. stream replay and thread status recover the same new-thread state.

If any check fails, stop writers again and restore the verified dump into an
empty replacement database. Do not introduce a request-time old-graph
fallback.
