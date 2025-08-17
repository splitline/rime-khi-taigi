local function autocap_filter(input, env)
    local ctx = env.engine.context
    local code = ctx.input -- Raw input code
    local code_len = #code
    local lomaji_mode = ctx:get_option("lomaji")

    local toned_map = {
        ["â"] = "Â", ["ê"] = "Ê", ["î"] = "Î", ["ô"] = "Ô", ["û"] = "Û",
        ["á"] = "Á", ["é"] = "É", ["í"] = "Í", ["ó"] = "Ó", ["ú"] = "Ú", ["ḿ"] = "Ḿ", ["ń"] = "Ń",
        ["à"] = "À", ["è"] = "È", ["ì"] = "Ì", ["ò"] = "Ò", ["ù"] = "Ù", ["ǹ"] = "Ǹ",
        ["ā"] = "Ā", ["ē"] = "Ē", ["ī"] = "Ī", ["ō"] = "Ō", ["ū"] = "Ū",
        ["ǎ"] = "Ǎ", ["ě"] = "Ě", ["ǐ"] = "Ǐ", ["ǒ"] = "Ǒ", ["ǔ"] = "Ǔ", ["ň"] = "Ň",
        ["ű"] = "Ű", ["ő"] = "Ő",
        ["ⁿ"] = "ᴺ"
    }

    local function convert_to_uppercase(text, mode)
        if mode == "all" then
            local result = text:upper()
            for lower, upper in pairs(toned_map) do
                result = result:gsub(lower, upper)
            end
            return result
        elseif mode == "first" then
            for lower, upper in pairs(toned_map) do
                if text:find("^" .. lower) then
                    return text:gsub("^" .. lower, upper)
                end
            end
            return text:gsub("^(%a)", string.upper)
        end
        return text
    end

    -- Determine capitalization mode
    local cap_mode = "none"
    if code:match("^%u%d?[ %-]*%u") then
        cap_mode = "all" -- At least the first two letters are uppercase -> All uppercase
    elseif code:match("^%u") then
        cap_mode = "first" -- Only the first letter is uppercase -> First letter uppercase
    end

    -- If code length is 1 or input does not start with an uppercase letter, no conversion is performed
    if code_len == 1 or cap_mode == "none" then
        for cand in input:iter() do
            yield(cand)
        end
        return
    end

    local pure_code = code:gsub("[%s%p]", ""):lower()

    for cand in input:iter() do
        local text = cand.text
        local pure_text = text:gsub("[%s%p]", "")

        -- Simplified skip logic - only skip if contains spaces
        local should_skip = text:match("%s") or -- 包含空格
             (cand.type ~= "completion" and pure_code ~= pure_text:lower()) -- 非补全项且编码与词不匹配 (如 PS -> Photoshop)

        -- if should_skip and not lomaji_mode then
        --     yield(cand)
        -- else
            local new_text = convert_to_uppercase(text, cap_mode)
            if new_text and new_text ~= text then
                local _cand = Candidate(cand.type, 0, code_len, new_text, cand.comment)
                _cand.preedit = cand.preedit
                yield(_cand)
            else
                yield(cand)
            end
        -- end
    end
end

return autocap_filter