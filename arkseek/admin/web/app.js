/* All user-controlled content is inserted as text, never HTML. */
(() => {
  'use strict';
  const content = document.getElementById('content');
  const error = document.getElementById('error');
  const state = {view:'overview', mode:null, q:'', channel:'all', offset:0, selected:null, detail:null, before:null, status:'all', hours:24, log:null, generation:0};
  const labels = {overview:['运行概览','查看系统状态与最近请求'],sessions:['会话记录','群聊按会话查看，成员身份分别保留'],users:['用户与记忆','核对个人画像、原始来源和生效状态'],monitor:['请求监测','区分排队、处理与发送，查看包含排队的等待时间'],logs:['运行日志','查看最近日志片段，辅助定位运行问题'],roadmap:['后续规划','已记录后续阶段，本版提供只读查看']};
  const statuses = {queued:'排队中',processing:'处理中',sending:'发送中',sent:'已发送',send_failed:'发送失败',processing_failed:'处理失败 · 错误提示已发送',failed:'失败',interrupted:'被中断',unknown:'状态未确认'};
  const channels = {group:'群聊',private:'私聊',cli:'本机对话'};
  function el(tag, attrs={}, ...children) {
    const n = document.createElement(tag);
    for(const [k,v] of Object.entries(attrs)) {
      if(k==='class') n.className=v;
      else if(k.startsWith('on')) n.addEventListener(k.slice(2).toLowerCase(),v);
      else if(k==='value') n.value=v;
      else if(k==='disabled') n.disabled=Boolean(v);
      else n.setAttribute(k,v);
    }
    for(const child of children.flat()) if(child!==null && child!==undefined) n.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return n;
  }
  const btn = (text, fn, cls='') => el('button',{type:'button',class:cls,onClick:fn},text);
  const muted = text => el('span',{class:'muted'},text);
  const time = value => value ? new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date(value)) : '未记录';
  const duration = value => value===null || value===undefined ? '未记录' : (value/1000).toFixed(1)+' 秒';
  function pill(status) { return el('span',{class:'pill '+(status==='sent'?'ok':(['queued','processing','sending'].includes(status)?'active':(['failed','send_failed','processing_failed','interrupted'].includes(status)?'bad':'')))},statuses[status]||status); }
  async function api(resource, params={}) {
    const url = new URL('/api/'+resource,location.origin);
    for(const [k,v] of Object.entries(params)) if(v!==null && v!==undefined && v!=='') url.searchParams.set(k,v);
    const response=await fetch(url,{cache:'no-store'});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||'请求失败');
    return data;
  }
  function showError(exc) { error.textContent=exc.message||String(exc); error.hidden=false; }
  function panel(title,...children) { return el('section',{class:'panel'},el('h2',{},title),...children); }
  function stat(title,value,note) { return el('div',{class:'stat'},el('span',{},title),el('strong',{},value),el('span',{},note)); }
  function pager(data) {
    return el('div',{class:'pager'},el('button',{type:'button',disabled:state.offset===0,onClick:()=>{state.offset=Math.max(0,state.offset-50);load();}},'上一页'),muted(`共 ${data.total} 条`),el('button',{type:'button',disabled:state.offset+50>=data.total,onClick:()=>{state.offset+=50;load();}},'下一页'));
  }
  function searchForm(extra=[]) {
    const input=el('input',{type:'search',value:state.q,placeholder:state.view==='logs'?'在最近日志片段中搜索':'搜索内容或昵称',id:'page-search',maxLength:'200'});
    return el('form',{class:'toolbar',onSubmit:event=>{event.preventDefault();state.q=input.value;state.offset=0;state.before=null;load();}},el('div',{class:'field grow'},el('label',{for:'page-search'},'搜索'),input),...extra,el('button',{type:'submit',class:'primary'},'搜索'));
  }
  function selectField(label,id,options,value,fn) {
    const select=el('select',{id,onChange:event=>fn(event.target.value)},options.map(([key,text])=>el('option',{value:key},text)));
    select.value=value;
    return el('div',{class:'field'},el('label',{for:id},label),select);
  }
  function go(view,selected=null) {
    Object.assign(state,{view,selected,q:'',offset:0,before:null,detail:null,generation:state.generation+1});
    location.hash=view;load();
  }
  async function openConversation(id) {go('sessions',id);}
  async function openUser(id) {go('users',id);}
  function requestTable(items) {
    const table=el('table',{},el('thead',{},el('tr',{},['接收时间','会话','状态','排队','总等待','详情'].map(t=>el('th',{},t)))));
    const body=el('tbody');
    for(const r of items) {
      const show=()=>{state.detail=r;const target=document.getElementById('request-detail');if(target){target.replaceChildren(requestDetail(r));target.scrollIntoView({block:'nearest',behavior:'smooth'});}};
      body.append(el('tr',{class:'request'},el('td',{},time(r.received_at)),el('td',{},btn(r.conversation_name,()=>openConversation(r.conversation_id),'link-button')),el('td',{},pill(r.status)),el('td',{class:'num'},duration(r.queue_ms)),el('td',{class:'num'},duration(r.total_ms)),el('td',{},btn('查看',show,'link-button'))));
    }
    table.append(body);
    return items.length?el('div',{class:'table-wrap'},table):el('p',{class:'empty'},'当前范围没有请求记录');
  }
  function requestDetail(r) {
    const section=panel('请求详情',el('div',{class:'meta'},pill(r.status),muted(time(r.received_at))),el('p',{},'排队 '+duration(r.queue_ms)+' · 接收至结束 '+duration(r.total_ms)),btn('查看该会话',()=>openConversation(r.conversation_id),'link-button'));
    if(r.error_kind)section.append(el('p',{class:'bad-text'},'异常类型：'+r.error_kind));
    if(r.status==='unknown')section.append(el('p',{class:'notice'},'对应进程没有近期心跳，无法确认此请求最终结果。最后记录的阶段：'+(statuses[r.recorded_status]||r.recorded_status)));
    if(r.spans.length) {
      section.append(el('p',{class:'muted'},'以下步骤按处理阶段起点对齐；并行步骤会重叠，不能将耗时相加作为总等待。'));
      const max=Math.max(...r.spans.map(s=>s.start_ms+s.duration_ms),1);
      for(const s of r.spans) {
        const mark=el('span',{class:'segment'});mark.style.left=(s.start_ms/max*100)+'%';mark.style.width=Math.max(s.duration_ms/max*100,.5)+'%';
        section.append(el('div',{class:'timeline-row'},el('span',{},s.name+(s.task?' · '+s.task:'')),el('div',{class:'rail',role:'img','aria-label':`${s.name}，开始 ${duration(s.start_ms)}，耗时 ${duration(s.duration_ms)}`},mark),muted(duration(s.duration_ms))));
      }
    } else section.append(el('p',{class:'muted'},'此请求没有分步计时记录。'));
    section.append(el('details',{},el('summary',{},'技术标识'),el('p',{class:'details-id'},r.request_id)));
    return section;
  }
  async function overview() {
    const d=await api('overview');state.mode=d.mode;
    document.getElementById('mode').textContent=d.mode==='demo'?'演示模式 · 全部为虚构数据':'本机数据 · 只读接入';
    const nodes=[];
    if(d.mode==='demo')nodes.push(el('div',{class:'notice'},'当前展示虚构样例，便于体验后台布局与操作；没有读取真实聊天和日志。'));
    nodes.push(el('div',{class:'stats'},stat('会话数',d.sessions??'—','全部历史会话'),stat('消息数',d.messages??'—','用户和助手消息条数'),stat('画像条目',d.profile_facts??'—','包含被保护规则屏蔽的原存储项'),stat('最近消息',d.last_message_at?time(d.last_message_at):'—','北京时间')));
    for(const notice of d.notice)nodes.push(el('p',{class:'notice'},notice));
    if(d.monitor) {
      if(state.detail)state.detail=await api('request',{id:state.detail.request_id});
      const live=d.monitor.runtimes.some(r=>r.heartbeat_live);
      const runtime=panel('运行状态',el('div',{class:'meta'},el('span',{class:'pill '+(live?'ok':'')},live?'监测进程心跳正常':'未检测到近期进程心跳'),muted('最近24小时')),
        el('p',{},`排队 ${d.monitor.summary.queued} · 处理 ${d.monitor.summary.processing} · 发送 ${d.monitor.summary.sending} · 失败 ${d.monitor.summary.failed}`),
        el('p',{class:'muted'},'进程心跳不等于QQ当前连接正常。没有心跳也可能是旧版机器人尚未接入监测。'));
      const dropped=d.monitor.runtimes.reduce((n,r)=>n+(r.dropped||0),0);
      if(dropped)runtime.append(el('p',{class:'bad-text'},`监测队列曾丢弃 ${dropped} 条事件，显示结果可能不完整。`));
      nodes.push(runtime,panel('最近请求',requestTable(d.monitor.items),btn('查看全部请求',()=>go('monitor'),'link-button')),el('div',{id:'request-detail'},state.detail?requestDetail(state.detail):null));
    }
    return nodes;
  }
  async function sessions() {
    const d=await api('sessions',{q:state.q,channel:state.channel,offset:state.offset});
    const list=el('div',{class:'list'});
    for(const s of d.items)list.append(el('button',{type:'button',class:'item '+(state.selected===s.id?'selected':''),onClick:()=>{state.selected=s.id;state.before=null;state.q='';load();}},el('div',{class:'item-title'},el('strong',{},s.name),el('span',{class:'pill'},channels[s.channel])),el('div',{class:'meta'},`${s.messages} 条消息 · ${time(s.last_at)}`),el('div',{class:'preview'},s.preview)));
    if(!d.items.length)list.append(el('p',{class:'empty'},'没有匹配的会话'));
    const detail=el('div');
    if(state.selected) {
      const c=await api('conversation',{id:state.selected,q:state.q,before:state.before});
      const p=panel(c.name,el('p',{class:'muted'},`${channels[c.channel]} · 共 ${c.total} 条消息 · 以下是机器人保存的问答`));
      if(c.channel==='group')p.append(el('p',{class:'muted'},'这里只包含机器人接收到的消息，不是QQ完整群聊。'));
      if(c.summary)p.append(el('details',{},el('summary',{},'查看会话摘要（话题索引）'),el('div',{class:'summary'},c.summary.summary),muted('更新 '+time(c.summary.updated_at)+' · 已合并至消息序号 '+c.summary.summarized_up_to)));
      if(c.has_more)p.append(btn('查看更早记录',()=>{state.before=c.next_before;load();}));
      if(state.before!==null)p.append(btn('返回最新记录',()=>{state.before=null;load();}));
      for(const m of c.items) {
        const name=m.role==='assistant'?'ArkSeek → '+m.speaker_name:m.speaker_name;
        const head=el('div',{class:'message-head'},el('strong',{},m.role==='assistant'?'ArkSeek':m.speaker_name),muted(time(m.created_at)));
        if(m.role==='assistant')head.append(muted('回复对象：'+m.speaker_name));
        if(m.person_id)head.append(btn('查看画像',()=>openUser(m.person_id),'link-button'));
        p.append(el('article',{class:'message','aria-label':name},head,el('div',{class:'message-body'},m.text)));
      }
      if(!c.items.length)p.append(el('p',{class:'empty'},'当前搜索没有匹配的消息'));
      p.append(el('details',{},el('summary',{},'会话技术标识'),el('p',{class:'details-id'},c.id)));
      detail.append(p);
    }else detail.append(panel('选择会话',el('p',{class:'empty'},'从会话列表选择群聊或私聊，查看消息、回复对象和会话摘要。')));
    return [searchForm([selectField('渠道','channel',Object.entries({all:'全部渠道',...channels}),state.channel,value=>{state.channel=value;state.offset=0;load();})]),el('div',{class:'split'},el('div',{},list,pager(d)),detail)];
  }
  async function users() {
    const d=await api('users',{q:state.q,offset:state.offset});
    const list=el('div',{class:'list'});
    for(const u of d.items)list.append(el('button',{type:'button',class:'item '+(state.selected===u.id?'selected':''),onClick:()=>{state.selected=u.id;load();}},el('div',{class:'item-title'},el('strong',{},u.name),el('span',{class:'pill'},channels[u.channel])),el('div',{class:'meta'},`${u.scope} · ${u.profile_count} 条画像`)));
    const detail=el('div');
    if(state.selected) {
      const u=await api('user',{id:state.selected});
      const p=panel(u.name,el('p',{class:'muted'},'同一个人在不同群或私聊中保留独立身份；本页不自动合并。'));
      for(const fact of u.profiles)p.append(el('article',{class:'fact'},el('div',{class:'fact-head'},el('strong',{},fact.key),el('span',{class:'pill '+(fact.effective?'ok':'')},fact.effective?'当前规则允许读取':'已被保护规则屏蔽')),el('div',{class:'fact-value'},fact.value),el('div',{class:'source'},'来源原话：'+(fact.source_text||'未记录')),muted('更新 '+time(fact.updated_at))));
      if(!u.profiles.length)p.append(el('p',{class:'empty'},'尚无存储画像'));
      p.append(el('h3',{},'相关会话'));
      for(const c of u.conversations)p.append(el('p',{},btn(c.name,()=>openConversation(c.id),'link-button')));
      p.append(el('details',{},el('summary',{},'身份技术标识'),el('p',{class:'details-id'},u.id)));
      detail.append(p);
    }else detail.append(panel('选择用户',el('p',{class:'empty'},'选择成员，核对画像值、抽取来源和保护规则。')));
    return [searchForm(),el('div',{class:'split'},el('div',{},list,d.items.length?pager(d):el('p',{class:'empty'},'没有匹配的用户')),detail)];
  }
  async function monitor() {
    const d=await api('monitor',{status:state.status,hours:state.hours,offset:state.offset});
    if(state.detail)state.detail=await api('request',{id:state.detail.request_id});
    const filters=el('div',{class:'toolbar'},selectField('时间范围','hours',[['1','最近1小时'],['24','最近24小时'],['168','最近7天']],String(state.hours),v=>{state.hours=Number(v);state.offset=0;load();}),selectField('请求状态','status',[['all','全部状态'],['queued','排队中'],['processing','处理中'],['sending','发送中'],['sent','已发送'],['failed','失败与中断'],['unknown','状态未确认']],state.status,v=>{state.status=v;state.offset=0;load();}));
    return [filters,el('div',{class:'stats'},stat('请求数',d.summary.total,`最近${d.hours}小时`),stat('成功发送',d.summary.sent,'由QQ发送接口返回判定'),stat('失败与中断',d.summary.failed,'包括处理异常和发送失败'),stat('等待中位数',duration(d.summary.median_ms),'包含排队；有结束时间的请求')),
      d.capped?el('p',{class:'notice'},'当前范围超过一万条，统计仅覆盖最近一万条请求。'):null,
      panel('请求记录',requestTable(d.items),pager(d)),el('div',{id:'request-detail'},state.detail?requestDetail(state.detail):null)];
  }
  async function logs() {
    const d=await api('logs');
    if(!state.log && d.items.length)state.log=d.items[0].name;
    const nodes=[searchForm([selectField('日志文件','log-file',d.items.map(f=>[f.name,f.name]),state.log||'',v=>{state.log=v;load();})])];
    if(state.log) {
      const l=await api('log',{name:state.log,q:state.q,limit:100});
      nodes.push(panel('最近日志片段',el('p',{class:'muted'},'最多读取文件末尾256KB、显示500行；搜索仅作用于该片段。常见密钥格式已遮盖，原日志仍按私有数据保存。'),el('pre',{class:'logtext'},l.lines.join('\n')||'没有匹配的日志行'),l.truncated?el('p',{class:'muted'},'部分更早日志未显示。'):null));
    }else nodes.push(panel('运行日志',el('p',{class:'empty'},'尚无控制台日志；机器人启动后会自动保存。')));
    return nodes;
  }
  function roadmap() {
    return [el('div',{class:'roadmap'},panel('第二阶段 · 记忆管理',el('p',{},'画像纠错、记忆停用和操作记录。写入需要处理正在运行的会话缓存、摘要与派生索引，并预览影响范围。')),
      panel('第三阶段 · 运行与配置',el('p',{},'机器人启停、配置维护和运行检查。支持明确的进程归属，避免后台误停其他实例。')),
      panel('第四阶段 · 知识库维护',el('p',{},'查看来源状态、管理项目副本、构建任务与索引重建。保留原始科研资料只读约定。')),
      panel('第一版的当前边界',el('p',{},'本机单管理员、只读查询。不读取未接入机器人的QQ消息；旧请求缺少的结构化监测不反推补造。实时QQ连接状态、画像变更历史和逐条检索证据尚未接入。')))];
  }
  async function load(quiet=false) {
    const generation=++state.generation;
    error.hidden=true;
    const label=labels[state.view]||labels.overview;
    document.getElementById('title').textContent=label[0];document.getElementById('subtitle').textContent=label[1];
    document.querySelectorAll('nav button').forEach(n=>n.setAttribute('aria-current',n.dataset.view===state.view?'page':'false'));
    if(!quiet)content.replaceChildren(el('p',{class:'empty'},'正在载入…'));
    try {
      const render={overview,sessions,users,monitor,logs,roadmap}[state.view];
      const nodes=await render();
      if(generation!==state.generation)return;
      content.replaceChildren(...nodes.filter(Boolean));
      document.getElementById('updated').textContent='更新 '+time(new Date().toISOString());
    }catch(exc){if(generation===state.generation){showError(exc);if(!quiet)content.replaceChildren(el('p',{class:'empty'},'当前数据暂不可用；可切换页面或刷新重试。'));}}
  }
  document.querySelectorAll('nav button').forEach(n=>n.addEventListener('click',()=>go(n.dataset.view)));
  document.getElementById('refresh').addEventListener('click',()=>load());
  document.getElementById('theme').addEventListener('click',()=>{const current=getComputedStyle(document.documentElement).colorScheme;const next=current==='dark'?'light':'dark';document.documentElement.style.colorScheme=next;try{localStorage.setItem('arkseek-admin-theme',next);}catch(_){};});
  try { const theme=localStorage.getItem('arkseek-admin-theme');if(['dark','light'].includes(theme))document.documentElement.style.colorScheme=theme; }catch(_){}
  setInterval(()=>{if(!document.hidden && document.getElementById('auto').checked && ['overview','monitor'].includes(state.view) && !document.activeElement.matches('input,select,textarea'))load(true);},10000);
  window.addEventListener('hashchange',()=>{const v=location.hash.slice(1);if(labels[v] && v!==state.view)go(v);});
  const initial=location.hash.slice(1);if(labels[initial])state.view=initial;
  // Obtain the dataset identity even when starting at a bookmarked subpage.
  api('overview').then(d=>{state.mode=d.mode;document.getElementById('mode').textContent=d.mode==='demo'?'演示模式 · 全部为虚构数据':'本机数据 · 只读接入';}).catch(showError);
  load();
})();
