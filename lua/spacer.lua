local F = {}

function F.func( input, env )
    local ctx = env.engine.context
    local commit_history = ctx.commit_history
    local composition = ctx.composition
    local last_text = nil
    -- \u0358 \u0300-\u030f
    local re = "[-a-zA-Z ̀-̏āēīōūâêîôûàèìòùǹáéíóúńḿⁿ]+"

    local commit_text = ctx:get_commit_text()
    local composition = ctx.composition
    local input_s = ctx.input
    local pos = composition:toSegmentation():get_confirmed_position()
    local sel_start = ctx:get_preedit().sel_start

    last_text = commit_history:latest_text()

    for cand in input:iter() do
        if cand.text:match( '^'..re..'$' ) then
            local text = (#cand.comment > 0) and cand.comment or cand.text
            if commit_text and #commit_text > 0
                and sel_start and sel_start > 0
                and commit_text:sub(0, sel_start):find(re..'$') then
                if input_s:sub(pos, pos) == " " then
                    text = " " .. text
                else
                    text = "-" .. text
                end
            elseif last_text and #last_text > 0
                and last_text:match( '^ ?'..re..'$' ) then
                text = " "..text
            end

            cand = cand:to_shadow_candidate('SP', text, cand.comment)
        end
        yield( cand )
    end
end

return F