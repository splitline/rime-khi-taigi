local F = {}

function F.func( input, env )
  local ctx = env.engine.context
  local is_lomaji = ctx:get_option("lomaji")

  for cand in input:iter() do
    if is_lomaji and (cand.type == "user_phrase" or cand.type == "phrase" or cand.type == "sentence") then
      yield(cand:to_shadow_candidate('Lomaji', cand.comment:gsub(" ","-"), cand.text))
    else
      yield(cand)
    end
  end
end

return F