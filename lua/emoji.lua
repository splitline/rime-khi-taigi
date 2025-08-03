local M = {}

-- Check if a string is an emoji (basic Unicode emoji ranges)
local function is_emoji(text)
  -- Emoji Unicode ranges (not exhaustive but covers common cases)
  local code = utf8.codepoint(text, 1)
  return (
    (code >= 0x1F300 and code <= 0x1FAFF) or -- General emoji blocks
    (code >= 0x2600 and code <= 0x27BF)     -- Misc symbols + Dingbats
  )
end

function M.func(input)
  local emoji_seen = false
  local emoji_buffer = {}

  local index = 1
  for cand in input:iter() do
    local text = cand.text
    if is_emoji(text) then
      if not emoji_seen then
        index = index + 1
        emoji_seen = true
        yield(cand)  -- keep the first emoji at its place
      else
        table.insert(emoji_buffer, cand)  -- store extra emojis
      end
    else
      index = index + 1
      emoji_seen = false
      yield(cand) -- non-emoji candidates stay untouched
    end
    if index >= 10 then break end
  end

  local next_cand = nil
  for cand in input:iter() do
    next_cand = cand
    break
  end

  if is_emoji(next_cand.text) then
    yield(next_cand)
  end

  -- Yield the rest of the emojis after position 10
  for _, cand in ipairs(emoji_buffer) do
    yield(cand)
  end

  if not is_emoji(next_cand.text) then
    yield(next_cand)
  end

  for cand in input:iter() do
      yield(cand)
  end
end

return M
