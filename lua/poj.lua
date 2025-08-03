local F = {}

function F.init(env)
  local config = env.engine.schema.config
  F.poj_pj = Projection()
  F.poj_pj:load(config:get_list('translator/__comment_format_poj'))
end

function F.func( input, env )
  local ctx = env.engine.context
  local to_poj = ctx:get_option("poj")
  for cand in input:iter() do
    if to_poj and cand.type ~= "completion" and cand.type ~= "table" then
      cand = cand:to_shadow_candidate(
        cand.type,
        F.poj_pj:apply(cand.text, true),
        F.poj_pj:apply(cand.comment, true)
      )
    end
    yield(cand)
  end
end

return F