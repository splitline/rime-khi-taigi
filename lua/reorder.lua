local M = {}

function M.init(env)
    local config = env.engine.schema.config
    M.count = 3
    M.idx = 6
end

function M.func(input)
    local i = 1
    local buffer = {}       -- store all candidates for further processing
    local oneCharExists = false
    local oneCharCandidates = {}
    local result = {}

    for cand in input:iter() do
        cand.comment = cand.comment
        table.insert(buffer, cand)
        if i <= 10 then
            if utf8.len(cand.text) == 1 then
                oneCharExists = true
                break
            end
        end
        i = i + 1
        if #buffer >= 500 then break end
    end

    -- If no 1-char word in top 10, find M.count of them
    if not oneCharExists then
        for _, cand in ipairs(buffer) do
            if utf8.len(cand.text) == 1 and not cand.text:find("[%a%d]") then
                table.insert(oneCharCandidates, cand)
                if #oneCharCandidates >= M.count then break end
            end
        end
    end

    -- Step 2: Reorder candidates
    -- Insert 1-char candidates at position M.idx if needed
    if not oneCharExists and #oneCharCandidates > 0 then
        -- Add candidates before the insertion point
        for i = 1, M.idx - 1 do
            if buffer[i] then
                table.insert(result, buffer[i])
            end
        end
        -- Insert the found 1-char candidates
        for _, cand in ipairs(oneCharCandidates) do
            table.insert(result, cand)
        end
        -- Add the rest of the candidates
        for i = M.idx, #buffer do
            table.insert(result, buffer[i])
        end
    else
        -- If no reordering is needed, result is the same as buffer
        result = buffer
    end

    -- Yield final result
    for _, cand in ipairs(result) do
        yield(cand)
    end

    -- This part handles cases where the input iterator has more items
    -- than the buffer limit (500), ensuring they are not lost.
    for cand in input:iter() do
        yield(cand)
    end
end

return M
