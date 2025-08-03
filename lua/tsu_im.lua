local F = {}

function F.init(env)
  local config = env.engine.schema.config
  F.poj_pj = Projection()
  F.preedit_format = Projection()
  F.preedit_format:load(config:get_list('translator/preedit_format'))
end

function F.func( input, env )
  local ctx = env.engine.context
  local seg = ctx.composition:back()
  local _start, _end = seg._start, seg._end
  local text = ctx.input:sub(_start+1)
  local tsu_im = F.preedit_format:apply(text, true)
  local tsu_im_cand = Candidate("tsuim", _start, _end, tsu_im, "")  

  local index = 1
  for cand in input:iter() do
    if index == 10 then
      yield(tsu_im_cand)
    end
    yield(cand)
    index = index + 1
  end

  if index <= 10 then
    yield(tsu_im_cand)
  end
end

return F