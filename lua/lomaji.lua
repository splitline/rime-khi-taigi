local F = {}

function F.func( input, env )
  local ctx = env.engine.context


  for cand in input:iter() do
    if ctx:get_option("lomaji") then
      yield(cand:to_shadow_candidate('LO', cand.comment, cand.text))
    else
      yield(cand)
    end
  end
end

return F